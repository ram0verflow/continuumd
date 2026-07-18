//! A minimal MCP client (stdio transport, newline-delimited JSON-RPC 2.0).
//!
//! Servers are declared in ~/.aios/mcp.json:
//!   { "servers": { "files": { "command": "npx",
//!                             "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] } } }
//!
//! Their tools are offered to the model through the TOOL_NEEDED fault; the
//! daemon executes the call and hands the result back as context. Reads go
//! through a reader thread with a timeout so a wedged server costs a turn
//! ten seconds, never a hang.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

use serde_json::{json, Value};

pub struct ToolDef {
    pub server: String,
    pub name: String,
    pub description: String,
}

pub struct McpServer {
    pub name: String,
    child: Child,
    stdin: std::process::ChildStdin,
    lines: Receiver<String>,
    next_id: u64,
    pub tools: Vec<ToolDef>,
}

const TIMEOUT: Duration = Duration::from_secs(10);

impl McpServer {
    pub fn start(name: &str, command: &str, args: &[String]) -> Result<Self, String> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {command}: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;

        let (tx, lines) = channel::<String>();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        let mut server = McpServer {
            name: name.to_string(),
            child,
            stdin,
            lines,
            next_id: 0,
            tools: Vec::new(),
        };

        server.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "aios-daemon", "version": "0.1.0"}
            }),
        )?;
        server.notify("notifications/initialized", json!({}))?;

        let listed = server.request("tools/list", json!({}))?;
        for t in listed.pointer("/tools").and_then(|t| t.as_array()).unwrap_or(&Vec::new()) {
            server.tools.push(ToolDef {
                server: name.to_string(),
                name: t["name"].as_str().unwrap_or("").to_string(),
                description: t["description"].as_str().unwrap_or("").chars().take(140).collect(),
            });
        }
        Ok(server)
    }

    /// Call one tool; returns the concatenated text content of the result.
    pub fn call(&mut self, tool: &str, args: Value) -> Result<String, String> {
        let result = self.request("tools/call", json!({"name": tool, "arguments": args}))?;
        let mut out = String::new();
        for c in result.pointer("/content").and_then(|c| c.as_array()).unwrap_or(&Vec::new()) {
            if let Some(text) = c["text"].as_str() {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(text);
            }
        }
        if result.pointer("/isError").and_then(|e| e.as_bool()).unwrap_or(false) {
            return Err(if out.is_empty() { "tool error".into() } else { out });
        }
        Ok(out)
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        self.next_id += 1;
        let id = self.next_id;
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        writeln!(self.stdin, "{msg}").map_err(|e| e.to_string())?;
        self.stdin.flush().ok();

        let deadline = std::time::Instant::now() + TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Err(format!("{}: {method} timed out", self.name));
            }
            let line = self.lines.recv_timeout(remaining).map_err(|_| format!("{}: {method} timed out", self.name))?;
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            if v["id"].as_u64() == Some(id) {
                if let Some(err) = v.get("error") {
                    return Err(err["message"].as_str().unwrap_or("rpc error").to_string());
                }
                return Ok(v["result"].clone());
            }
            // Notifications and unrelated ids: skip.
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), String> {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        writeln!(self.stdin, "{msg}").map_err(|e| e.to_string())?;
        self.stdin.flush().ok();
        Ok(())
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

/// Boot every server declared in ~/.aios/mcp.json. Failures are logged and
/// skipped; a bad server must never take the daemon down.
pub fn start_all(config_path: &str) -> Vec<McpServer> {
    let Ok(raw) = std::fs::read_to_string(config_path) else { return Vec::new() };
    let Ok(cfg) = serde_json::from_str::<Value>(&raw) else {
        eprintln!("[mcp] {config_path} is not valid JSON; ignoring");
        return Vec::new();
    };
    let mut servers = Vec::new();
    for (name, s) in cfg["servers"].as_object().cloned().unwrap_or_default() {
        let command = s["command"].as_str().unwrap_or("").to_string();
        if command.is_empty() {
            continue;
        }
        let args: Vec<String> = s["args"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        match McpServer::start(&name, &command, &args) {
            Ok(srv) => {
                eprintln!("[mcp] {name}: {} tools", srv.tools.len());
                servers.push(srv);
            }
            Err(e) => eprintln!("[mcp] {name} failed: {e}"),
        }
    }
    servers
}

/// "server.tool {json}" -> (server, tool, args). Args default to {}.
pub fn parse_tool_request(rest: &str) -> Option<(String, String, Value)> {
    let rest = rest.trim();
    let (full_name, arg_str) = match rest.find(|c: char| c.is_whitespace()) {
        Some(pos) => (&rest[..pos], rest[pos..].trim()),
        None => (rest, ""),
    };
    let (server, tool) = full_name.split_once('.')?;
    let args = if arg_str.is_empty() {
        json!({})
    } else {
        serde_json::from_str(arg_str).unwrap_or(json!({}))
    };
    Some((server.to_string(), tool.to_string(), args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_request_parsing() {
        let (s, t, a) = parse_tool_request("files.read_file {\"path\": \"/tmp/x\"}").unwrap();
        assert_eq!((s.as_str(), t.as_str()), ("files", "read_file"));
        assert_eq!(a["path"], "/tmp/x");
        let (s, t, a) = parse_tool_request("calc.add").unwrap();
        assert_eq!((s.as_str(), t.as_str()), ("calc", "add"));
        assert_eq!(a, json!({}));
        assert!(parse_tool_request("no_dot_here {}").is_none());
    }
}
