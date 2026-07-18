//! Minimal, dependency-free Ollama client.
//!
//! Talks HTTP/1.1 to a local Ollama daemon over a raw `TcpStream`. We only ever
//! hit `127.0.0.1:11434`, so there is no TLS and no need to pull in `reqwest`.
//! Requests are sent with `Connection: close` and the whole response is read to
//! EOF, then de-chunked if necessary.

use serde_json::{json, Value};

use crate::http;

const PORT: u16 = 11434;

#[derive(Clone)]
pub struct Ollama {
    pub chat_model: String,
    pub embed_model: String,
}

#[derive(Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Base64 image payloads for vision models, in the exact field shape
    /// Ollama's chat API expects. None serializes to nothing, so text-only
    /// traffic is byte-identical to before.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        ChatMessage { role: role.to_string(), content: content.into(), images: None }
    }

    pub fn with_images(role: &str, content: impl Into<String>, images: Vec<String>) -> Self {
        ChatMessage { role: role.to_string(), content: content.into(), images: Some(images) }
    }
}

impl Ollama {
    pub fn new(chat_model: &str, embed_model: &str) -> Self {
        Ollama { chat_model: chat_model.to_string(), embed_model: embed_model.to_string() }
    }

    /// One non-streamed chat completion. Returns the assistant text.
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        num_ctx: usize,
        num_predict: usize,
    ) -> Result<String, String> {
        let body = json!({
            "model": self.chat_model,
            "messages": messages,
            "stream": false,
            "options": { "num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.0 }
        });
        let resp = self.post("/api/chat", &body)?;
        resp.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("unexpected chat response: {resp}"))
    }

    /// Embed a single string via the embedding model.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let body = json!({ "model": self.embed_model, "input": text });
        let resp = self.post("/api/embed", &body)?;
        // /api/embed returns {"embeddings": [[...]]}
        let arr = resp
            .get("embeddings")
            .and_then(|e| e.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("unexpected embed response: {resp}"))?;
        Ok(arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
    }

    /// True if the daemon answers and the models load.
    pub fn healthy(&self) -> bool {
        self.embed("ping").is_ok()
    }

    /// Streaming chat: calls `on_token` for every generated piece as it
    /// arrives, returns the full response text at the end. Reads Ollama's
    /// chunked NDJSON incrementally off the socket.
    pub fn chat_stream(
        &self,
        messages: &[ChatMessage],
        num_ctx: usize,
        num_predict: usize,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        use std::io::{Read, Write};
        let body = serde_json::to_vec(&json!({
            "model": self.chat_model,
            "messages": messages,
            "stream": true,
            "options": { "num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.0 }
        })).map_err(|e| e.to_string())?;

        let mut stream = std::net::TcpStream::connect(("127.0.0.1", PORT))
            .map_err(|e| format!("connect ollama: {e}"))?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(600))).ok();
        write!(
            stream,
            "POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        ).map_err(|e| e.to_string())?;
        stream.write_all(&body).map_err(|e| e.to_string())?;

        // Skip response headers, then de-chunk and split NDJSON lines as bytes
        // arrive. Each line is one JSON object with a message.content piece.
        let mut buf: Vec<u8> = Vec::new();
        let mut tmp = [0u8; 8192];
        let mut headers_done = false;
        let mut chunked = false;
        let mut payload: Vec<u8> = Vec::new();
        let mut line_start = 0usize;
        let mut full = String::new();

        loop {
            let n = stream.read(&mut tmp).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);

            if !headers_done {
                if let Some(pos) = http::find_subslice(&buf, b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                    if !head.contains(" 200") {
                        return Err(format!("ollama HTTP error: {}", head.lines().next().unwrap_or("")));
                    }
                    chunked = head.contains("transfer-encoding: chunked");
                    buf.drain(..pos + 4);
                    headers_done = true;
                } else {
                    continue;
                }
            }

            // Move whatever body bytes we have into the payload, de-chunking
            // greedily. Incomplete chunks stay in buf for the next read.
            if chunked {
                loop {
                    let Some(nl) = http::find_subslice(&buf, b"\r\n") else { break };
                    let size = usize::from_str_radix(String::from_utf8_lossy(&buf[..nl]).trim(), 16).unwrap_or(0);
                    if size == 0 {
                        buf.clear();
                        break;
                    }
                    if buf.len() < nl + 2 + size + 2 {
                        break; // chunk not fully here yet
                    }
                    payload.extend_from_slice(&buf[nl + 2..nl + 2 + size]);
                    buf.drain(..nl + 2 + size + 2);
                }
            } else {
                payload.append(&mut buf);
            }

            // Emit tokens for every complete NDJSON line.
            while let Some(rel) = payload[line_start..].iter().position(|&b| b == b'\n') {
                let line = String::from_utf8_lossy(&payload[line_start..line_start + rel]).to_string();
                line_start += rel + 1;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if let Some(piece) = v.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()) {
                        if !piece.is_empty() {
                            full.push_str(piece);
                            on_token(piece);
                        }
                    }
                    if v.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                        return Ok(full);
                    }
                }
            }
        }
        Ok(full)
    }

    fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        http::post_json(PORT, path, body)
    }
}
