//! Provider adapters: one trait, provider-independent message schema
//! (`aios::ollama::ChatMessage` — role + content), streaming via callback,
//! cooperative cancellation.
//!
//! Model switching preserves continuity by construction: memory never lives
//! in the model. The KV cache tier does not survive a switch; it is simply
//! not restored for a different model.
//!
//! Frontier models are not fine-tuned for the fault protocol; they get the
//! protocol in the system prompt (the kernel's SYSTEM_TEMPLATE states it)
//! and the kernel's soft-refusal detector catches "I don't have that"
//! phrasings. The local tuned model was trained on those exact bytes.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{json, Value};

use aios::ollama::ChatMessage;

use crate::state::{Keys, Settings};

#[allow(dead_code)] // capability flags: consumed by future clients (VS Code, CLI)
pub struct ProviderCaps {
    pub supports_system: bool,
    pub max_context: usize,
}

pub trait Provider: Send {
    /// "provider/model", for the inspector and the journal.
    fn label(&self) -> String;

    #[allow(dead_code)]
    fn caps(&self) -> ProviderCaps;

    /// Stream one completion. `on_token` receives each piece as it arrives;
    /// the full text is returned at the end. When `cancel` flips true the
    /// adapter stops reading (dropping the connection stops generation
    /// server-side) and returns what it has.
    fn chat_stream(
        &self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_token: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<String, String>;
}

/// Build the adapter the current settings ask for. Cheap; called per turn so
/// a settings change takes effect on the next message.
pub fn build(settings: &Settings, keys: &Keys) -> Result<Box<dyn Provider>, String> {
    match settings.provider.as_str() {
        "ollama" => Ok(Box::new(OllamaProvider {
            model: settings.model.clone(),
            num_ctx: settings.num_ctx,
        })),
        "claude" => {
            let key = keys
                .get("anthropic")
                .ok_or("no Anthropic key: put {\"anthropic\": \"sk-...\"} in ~/.aios/keys or set ANTHROPIC_API_KEY")?;
            Ok(Box::new(ClaudeProvider { model: settings.model.clone(), api_key: key.to_string() }))
        }
        "openai_compat" => Ok(Box::new(OpenAICompatProvider {
            base_url: settings.base_url.trim_end_matches('/').to_string(),
            model: settings.model.clone(),
            api_key: keys.get("openai").unwrap_or("").to_string(),
        })),
        "llamaserver" => Ok(Box::new(LlamaServerProvider { port: 8080 })),
        "bedrock" => Ok(Box::new(BedrockProvider {
            model_id: settings.model.clone(),
            region: crate::bedrock::default_region(),
        })),
        other => Err(format!("unknown provider '{other}'")),
    }
}

fn cancelled(c: &AtomicBool) -> bool {
    c.load(Ordering::Relaxed)
}

// --- Ollama -----------------------------------------------------------------

pub struct OllamaProvider {
    pub model: String,
    pub num_ctx: usize,
}

impl Provider for OllamaProvider {
    fn label(&self) -> String {
        format!("ollama/{}", self.model)
    }

    fn caps(&self) -> ProviderCaps {
        ProviderCaps { supports_system: true, max_context: self.num_ctx }
    }

    fn chat_stream(
        &self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_token: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        // Ollama wants raw base64 in the images field; ours arrive as data: URLs.
        let msgs: Vec<ChatMessage> = messages
            .iter()
            .map(|m| {
                let mut m2 = m.clone();
                if let Some(imgs) = &m.images {
                    m2.images = Some(
                        imgs.iter()
                            .map(|d| d.split_once(";base64,").map(|(_, b)| b.to_string()).unwrap_or_else(|| d.clone()))
                            .collect(),
                    );
                }
                m2
            })
            .collect();
        let body = json!({
            "model": self.model,
            "messages": msgs,
            "stream": true,
            "options": { "num_ctx": self.num_ctx, "num_predict": max_tokens, "temperature": temperature }
        });
        let resp = match ureq::post("http://127.0.0.1:11434/api/chat").send_json(body) {
            Ok(r) => r,
            // Text-only model, image attached: degrade gracefully — drop the
            // images and let the model say it couldn't see them, instead of
            // surfacing a raw HTTP 400 as the reply.
            Err(e) => {
                let msg = fmt_ureq(e);
                if !msg.contains("multimodal") {
                    return Err(msg);
                }
                let mut text_only: Vec<ChatMessage> = messages.to_vec();
                for m in &mut text_only {
                    if m.images.take().is_some() {
                        m.content.push_str(
                            "\n\n(Note: an image was attached, but the current model cannot see images. \
                             Mention that briefly and suggest switching to a vision model.)",
                        );
                    }
                }
                let body = json!({
                    "model": self.model,
                    "messages": text_only,
                    "stream": true,
                    "options": { "num_ctx": self.num_ctx, "num_predict": max_tokens, "temperature": temperature }
                });
                ureq::post("http://127.0.0.1:11434/api/chat").send_json(body).map_err(fmt_ureq)?
            }
        };
        let reader = BufReader::new(resp.into_reader());
        let mut full = String::new();
        for line in reader.lines() {
            if cancelled(cancel) {
                return Ok(full);
            }
            let line = line.map_err(|e| e.to_string())?;
            if line.trim().is_empty() {
                continue;
            }
            let v: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(piece) = v.pointer("/message/content").and_then(|c| c.as_str()) {
                if !piece.is_empty() {
                    full.push_str(piece);
                    on_token(piece);
                }
            }
            if v.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                break;
            }
        }
        Ok(full)
    }
}

// --- Claude (Anthropic Messages API) ------------------------------------------

pub struct ClaudeProvider {
    pub model: String,
    pub api_key: String,
}

impl Provider for ClaudeProvider {
    fn label(&self) -> String {
        format!("claude/{}", self.model)
    }

    fn caps(&self) -> ProviderCaps {
        ProviderCaps { supports_system: true, max_context: 200_000 }
    }

    fn chat_stream(
        &self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_token: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        let (system, turns) = split_system(messages);
        let body = json!({
            "model": self.model,
            "system": system,
            "messages": turns,
            "max_tokens": max_tokens.max(1),
            "temperature": temperature,
            "stream": true,
        });
        let resp = ureq::post("https://api.anthropic.com/v1/messages")
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", "2023-06-01")
            .send_json(body)
            .map_err(fmt_ureq)?;
        let reader = BufReader::new(resp.into_reader());
        let mut full = String::new();
        for line in reader.lines() {
            if cancelled(cancel) {
                return Ok(full);
            }
            let line = line.map_err(|e| e.to_string())?;
            let Some(data) = line.strip_prefix("data: ") else { continue };
            let v: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            match v.get("type").and_then(|t| t.as_str()) {
                Some("content_block_delta") => {
                    if let Some(piece) = v.pointer("/delta/text").and_then(|t| t.as_str()) {
                        if !piece.is_empty() {
                            full.push_str(piece);
                            on_token(piece);
                        }
                    }
                }
                Some("error") => {
                    return Err(format!("claude: {}", v.pointer("/error/message").and_then(|m| m.as_str()).unwrap_or("stream error")));
                }
                Some("message_stop") => break,
                _ => {}
            }
        }
        Ok(full)
    }
}

/// Anthropic wants system as a top-level field and strictly alternating
/// user/assistant turns; fold system-role messages into the system string,
/// merge consecutive same-role messages, and expand attached images into
/// content blocks.
fn split_system(messages: &[ChatMessage]) -> (String, Vec<Value>) {
    let mut system = String::new();
    let mut turns: Vec<(String, String, Vec<String>)> = Vec::new();
    for m in messages {
        if m.role == "system" {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&m.content);
            continue;
        }
        let imgs = m.images.clone().unwrap_or_default();
        match turns.last_mut() {
            Some((role, content, images)) if *role == m.role => {
                content.push_str("\n\n");
                content.push_str(&m.content);
                images.extend(imgs);
            }
            _ => turns.push((m.role.clone(), m.content.clone(), imgs)),
        }
    }
    if turns.first().map(|(r, _, _)| r == "assistant").unwrap_or(false) {
        turns.insert(0, ("user".into(), "(continuing)".into(), Vec::new()));
    }
    let turns = turns
        .into_iter()
        .map(|(role, content, images)| {
            if images.is_empty() {
                json!({"role": role, "content": content})
            } else {
                let mut blocks: Vec<Value> = images
                    .iter()
                    .filter_map(|d| {
                        let (head, b64) = d.split_once(";base64,")?;
                        let mime = head.strip_prefix("data:").unwrap_or("image/png");
                        Some(json!({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}))
                    })
                    .collect();
                blocks.push(json!({"type": "text", "text": content}));
                json!({"role": role, "content": blocks})
            }
        })
        .collect();
    (system, turns)
}

// --- OpenAI-compatible ------------------------------------------------------------

/// One adapter covers OpenAI, LM Studio, vLLM, OpenRouter, Gemini's OpenAI
/// endpoint. Base URL and key are settings.
pub struct OpenAICompatProvider {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

impl Provider for OpenAICompatProvider {
    fn label(&self) -> String {
        format!("openai_compat/{}", self.model)
    }

    fn caps(&self) -> ProviderCaps {
        ProviderCaps { supports_system: true, max_context: 128_000 }
    }

    fn chat_stream(
        &self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_token: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        // OpenAI-style content arrays only where images demand them.
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| match &m.images {
                Some(imgs) if !imgs.is_empty() => {
                    let mut parts = vec![json!({"type": "text", "text": m.content})];
                    for d in imgs {
                        parts.push(json!({"type": "image_url", "image_url": {"url": d}}));
                    }
                    json!({"role": m.role, "content": parts})
                }
                _ => json!({"role": m.role, "content": m.content}),
            })
            .collect();
        let body = json!({
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens.max(1),
            "temperature": temperature,
            "stream": true,
        });
        let mut req = ureq::post(&format!("{}/chat/completions", self.base_url));
        if !self.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        let resp = req.send_json(body).map_err(fmt_ureq)?;
        let reader = BufReader::new(resp.into_reader());
        let mut full = String::new();
        for line in reader.lines() {
            if cancelled(cancel) {
                return Ok(full);
            }
            let line = line.map_err(|e| e.to_string())?;
            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data.trim() == "[DONE]" {
                break;
            }
            let v: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(piece) = v.pointer("/choices/0/delta/content").and_then(|c| c.as_str()) {
                if !piece.is_empty() {
                    full.push_str(piece);
                    on_token(piece);
                }
            }
        }
        Ok(full)
    }
}

// --- llama-server (KV-paging backend) -------------------------------------------

/// Non-streaming for now, same as the prototype: llama-server is the KV
/// save/restore path, and its SSE parsing is not wired yet.
pub struct LlamaServerProvider {
    pub port: u16,
}

impl Provider for LlamaServerProvider {
    fn label(&self) -> String {
        format!("llamaserver:{}", self.port)
    }

    fn caps(&self) -> ProviderCaps {
        ProviderCaps { supports_system: true, max_context: 8192 }
    }

    fn chat_stream(
        &self,
        messages: &[ChatMessage],
        max_tokens: usize,
        _temperature: f32,
        on_token: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        if cancelled(cancel) {
            return Ok(String::new());
        }
        let server = aios::llamaserver::LlamaServer::new(self.port);
        let full = server.chat(messages, max_tokens)?;
        if !cancelled(cancel) {
            on_token(&full);
        }
        Ok(full)
    }
}

// --- Claude on Bedrock (the user's own AWS account) ---------------------------

/// Non-streaming, like llama-server: Bedrock's stream uses AWS event-stream
/// binary framing, deferred until the plain path proves out. Credentials
/// resolve through the AWS CLI, so `aws login` is the only setup.
pub struct BedrockProvider {
    pub model_id: String,
    pub region: String,
}

impl Provider for BedrockProvider {
    fn label(&self) -> String {
        format!("bedrock/{}", self.model_id)
    }

    fn caps(&self) -> ProviderCaps {
        ProviderCaps { supports_system: true, max_context: 200_000 }
    }

    fn chat_stream(
        &self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_token: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        if cancelled(cancel) {
            return Ok(String::new());
        }
        let (system, turns) = converse_messages(messages);
        let full = crate::bedrock::converse(&self.region, &self.model_id, &system, &turns, max_tokens, temperature)?;
        if !cancelled(cancel) {
            on_token(&full);
        }
        Ok(full)
    }
}

/// Same folding as the Anthropic API, in Converse content shapes.
fn converse_messages(messages: &[ChatMessage]) -> (String, Vec<Value>) {
    let mut system = String::new();
    let mut turns: Vec<(String, String, Vec<String>)> = Vec::new();
    for m in messages {
        if m.role == "system" {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&m.content);
            continue;
        }
        let imgs = m.images.clone().unwrap_or_default();
        match turns.last_mut() {
            Some((role, content, images)) if *role == m.role => {
                content.push_str("\n\n");
                content.push_str(&m.content);
                images.extend(imgs);
            }
            _ => turns.push((m.role.clone(), m.content.clone(), imgs)),
        }
    }
    if turns.first().map(|(r, _, _)| r == "assistant").unwrap_or(false) {
        turns.insert(0, ("user".into(), "(continuing)".into(), Vec::new()));
    }
    let turns = turns
        .into_iter()
        .map(|(role, content, images)| {
            let mut blocks: Vec<Value> = images
                .iter()
                .filter_map(|d| {
                    let (head, b64) = d.split_once(";base64,")?;
                    let format = match head.strip_prefix("data:").unwrap_or("") {
                        "image/jpeg" => "jpeg",
                        "image/webp" => "webp",
                        "image/gif" => "gif",
                        _ => "png",
                    };
                    Some(json!({"image": {"format": format, "source": {"bytes": b64}}}))
                })
                .collect();
            blocks.push(json!({"text": content}));
            json!({"role": role, "content": blocks})
        })
        .collect();
    (system, turns)
}

fn fmt_ureq(e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .or(v.pointer("/error"))
                        .and_then(|m| m.as_str().map(|s| s.to_string()))
                })
                .unwrap_or_else(|| body.chars().take(300).collect());
            format!("HTTP {code}: {msg}")
        }
        ureq::Error::Transport(t) => format!("transport: {t}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_system_folds_and_merges() {
        let msgs = vec![
            ChatMessage::new("system", "protocol"),
            ChatMessage::new("system", "[PREVIOUS CONTEXT] earlier"),
            ChatMessage::new("user", "a"),
            ChatMessage::new("user", "b"),
            ChatMessage::new("assistant", "c"),
            ChatMessage::new("user", "d"),
        ];
        let (system, turns) = split_system(&msgs);
        assert!(system.contains("protocol") && system.contains("earlier"));
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0]["content"], "a\n\nb");
        assert_eq!(turns[0]["role"], "user");
        assert_eq!(turns[2]["role"], "user");
    }
}
