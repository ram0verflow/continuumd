//! Web search for the WEB_NEEDED fault. Brave's JSON API when a key is on
//! file ({"brave": "..."} in ~/.aios/keys), otherwise a keyless DuckDuckGo
//! HTML fallback. Results are handed back to the model as a plain text
//! block; the daemon never follows links itself.

use serde_json::Value;

pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub fn search(query: &str, brave_key: Option<&str>) -> Result<Vec<Hit>, String> {
    match brave_key {
        Some(key) => brave(query, key),
        None => ddg(query),
    }
}

pub fn provider_name(brave_key: Option<&str>) -> &'static str {
    if brave_key.is_some() { "brave" } else { "duckduckgo" }
}

/// Render hits into the block the model reads after a web fault.
pub fn render_block(query: &str, hits: &[Hit]) -> String {
    let mut out = format!("[WEB RESULTS: {query}]\n");
    for (i, h) in hits.iter().enumerate() {
        out.push_str(&format!("{}. {} — {}\n   {}\n", i + 1, h.title, h.url, h.snippet));
    }
    out.push_str(
        "\nAnswer the user's question using these results. Mention the source \
         link in plain text when it matters. If the results don't answer it, say so.",
    );
    out
}

fn brave(query: &str, key: &str) -> Result<Vec<Hit>, String> {
    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}&count=6",
        urlencode(query)
    );
    let resp = ureq::get(&url)
        .set("X-Subscription-Token", key)
        .set("Accept", "application/json")
        .timeout(std::time::Duration::from_secs(10))
        .call()
        .map_err(|e| format!("brave: {e}"))?;
    let v: Value = resp.into_json().map_err(|e| e.to_string())?;
    let mut hits = Vec::new();
    for r in v.pointer("/web/results").and_then(|r| r.as_array()).unwrap_or(&Vec::new()) {
        hits.push(Hit {
            title: strip_tags(r["title"].as_str().unwrap_or("")),
            url: r["url"].as_str().unwrap_or("").to_string(),
            snippet: strip_tags(r["description"].as_str().unwrap_or("")),
        });
        if hits.len() >= 6 {
            break;
        }
    }
    Ok(hits)
}

/// Keyless fallback: DuckDuckGo's HTML endpoint, parsed by string scanning.
/// Fragile by nature; good enough until a Brave key shows up.
fn ddg(query: &str) -> Result<Vec<Hit>, String> {
    let resp = ureq::post("https://html.duckduckgo.com/html/")
        .set("Content-Type", "application/x-www-form-urlencoded")
        .set("User-Agent", "Mozilla/5.0 (Macintosh) aios-daemon")
        .timeout(std::time::Duration::from_secs(10))
        .send_string(&format!("q={}", urlencode(query)))
        .map_err(|e| format!("ddg: {e}"))?;
    let html = resp.into_string().map_err(|e| e.to_string())?;

    let mut hits = Vec::new();
    for chunk in html.split("result__a").skip(1) {
        let Some(href) = between(chunk, "href=\"", "\"") else { continue };
        let Some(title_raw) = between(chunk, ">", "</a>") else { continue };
        let snippet = chunk
            .split_once("result__snippet")
            .and_then(|(_, rest)| between(rest, ">", "</a>"))
            .unwrap_or_default();
        let url = decode_ddg_href(&href);
        if url.is_empty() {
            continue;
        }
        hits.push(Hit {
            title: strip_tags(&decode_entities(&title_raw)),
            url,
            snippet: strip_tags(&decode_entities(&snippet)),
        });
        if hits.len() >= 6 {
            break;
        }
    }
    if hits.is_empty() {
        return Err("no results parsed".into());
    }
    Ok(hits)
}

/// DDG links come as //duckduckgo.com/l/?uddg=<urlencoded-target>&...
fn decode_ddg_href(href: &str) -> String {
    if let Some(pos) = href.find("uddg=") {
        let tail = &href[pos + 5..];
        let end = tail.find('&').unwrap_or(tail.len());
        return urldecode(&tail[..end]);
    }
    if href.starts_with("http") {
        return href.to_string();
    }
    String::new()
}

fn between(s: &str, a: &str, b: &str) -> Option<String> {
    let start = s.find(a)? + a.len();
    let end = s[start..].find(b)? + start;
    Some(s[start..end].to_string())
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.trim().to_string()
}

fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 2;
                    }
                    Err(_) => out.push(bytes[i]),
                }
            }
            b'+' => out.push(b' '),
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddg_redirect_decoding() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc";
        assert_eq!(decode_ddg_href(href), "https://example.com/page");
        assert_eq!(decode_ddg_href("https://direct.example"), "https://direct.example");
        assert_eq!(decode_ddg_href("/ad_click"), "");
    }

    #[test]
    fn tag_stripping_and_block() {
        assert_eq!(strip_tags("a <b>bold</b> claim"), "a bold claim");
        let block = render_block("rust 1.80", &[Hit { title: "T".into(), url: "https://u".into(), snippet: "S".into() }]);
        assert!(block.contains("[WEB RESULTS: rust 1.80]"));
        assert!(block.contains("1. T — https://u"));
    }
}
