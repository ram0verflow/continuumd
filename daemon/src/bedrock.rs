//! Claude on AWS Bedrock, self-hosted-account style.
//!
//! Credentials come from `aws configure export-credentials`, which resolves
//! whatever the AWS CLI is set up with (SSO via `aws login`, static keys,
//! env vars) into a temporary key set. The daemon signs requests itself
//! (SigV4) and calls the Converse API. Non-streaming for now: Bedrock
//! streams in AWS event-stream binary framing, which is not worth parsing
//! until the rest proves out; replies arrive whole, like the llama-server
//! path.

use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub struct AwsCreds {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
    pub expires_at: Option<u64>,
}

static CRED_CACHE: Mutex<Option<AwsCreds>> = Mutex::new(None);

/// Resolve credentials through the AWS CLI, cached until near expiry.
pub fn credentials() -> Result<AwsCreds, String> {
    {
        let cache = CRED_CACHE.lock().unwrap();
        if let Some(c) = cache.as_ref() {
            let fresh = c.expires_at.map_or(true, |exp| now_secs() + 120 < exp);
            if fresh {
                return Ok(AwsCreds {
                    access_key: c.access_key.clone(),
                    secret_key: c.secret_key.clone(),
                    session_token: c.session_token.clone(),
                    expires_at: c.expires_at,
                });
            }
        }
    }
    // --format process is the widely supported JSON output (--format json
    // only exists in newer CLI builds); the fields are identical.
    let out = Command::new("aws")
        .args(["configure", "export-credentials", "--format", "process"])
        .output()
        .map_err(|e| format!("aws cli not found: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "aws credentials unavailable ({}). Run `aws login` and try again.",
            err.lines().next().unwrap_or("unknown").trim()
        ));
    }
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())?;
    let creds = AwsCreds {
        access_key: v["AccessKeyId"].as_str().unwrap_or("").to_string(),
        secret_key: v["SecretAccessKey"].as_str().unwrap_or("").to_string(),
        session_token: v["SessionToken"].as_str().map(String::from),
        expires_at: v["Expiration"].as_str().and_then(parse_iso8601),
    };
    if creds.access_key.is_empty() {
        return Err("aws returned empty credentials".into());
    }
    *CRED_CACHE.lock().unwrap() = Some(AwsCreds {
        access_key: creds.access_key.clone(),
        secret_key: creds.secret_key.clone(),
        session_token: creds.session_token.clone(),
        expires_at: creds.expires_at,
    });
    Ok(creds)
}

pub fn default_region() -> String {
    if let Ok(r) = std::env::var("AWS_REGION") {
        if !r.is_empty() {
            return r;
        }
    }
    Command::new("aws")
        .args(["configure", "get", "region"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "us-east-1".to_string())
}

static PROFILE_CACHE: Mutex<Option<(u64, Vec<String>)>> = Mutex::new(None);

/// Claude inference profiles visible to this account in this region,
/// cached for two minutes so the model switcher opens without a CLI call.
pub fn list_claude_profiles(region: &str) -> Vec<String> {
    {
        let cache = PROFILE_CACHE.lock().unwrap();
        if let Some((at, profiles)) = cache.as_ref() {
            if now_secs() < at + 120 {
                return profiles.clone();
            }
        }
    }
    let profiles = list_claude_profiles_uncached(region);
    if !profiles.is_empty() {
        *PROFILE_CACHE.lock().unwrap() = Some((now_secs(), profiles.clone()));
    }
    profiles
}

fn list_claude_profiles_uncached(region: &str) -> Vec<String> {
    let out = Command::new("aws")
        .args(["bedrock", "list-inference-profiles", "--region", region, "--output", "json"])
        .output();
    let Ok(out) = out else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    let Ok(v) = serde_json::from_slice::<Value>(&out.stdout) else { return Vec::new() };
    // Chat-capable families only; upscalers and embedders don't converse.
    const CHAT_FAMILIES: [&str; 5] = ["anthropic", "amazon.nova", "meta.llama", "mistral", "deepseek"];
    v["inferenceProfileSummaries"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| p["inferenceProfileId"].as_str())
                .filter(|id| CHAT_FAMILIES.iter().any(|f| id.contains(f)))
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

/// One non-streaming Converse call. `messages` is Converse-shaped JSON.
pub fn converse(
    region: &str,
    model_id: &str,
    system: &str,
    messages: &[Value],
    max_tokens: usize,
    temperature: f32,
) -> Result<String, String> {
    let creds = credentials()?;
    let host = format!("bedrock-runtime.{region}.amazonaws.com");
    let path = format!("/model/{}/converse", uri_encode(model_id));
    // SigV4 wants the canonical path segment-encoded TWICE for every service
    // but S3: the wire carries %3A, the signature is computed over %253A.
    let canonical_path = path.replace('%', "%25");
    let body = json!({
        "system": [{"text": system}],
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens.max(1), "temperature": temperature},
    });
    let payload = serde_json::to_vec(&body).map_err(|e| e.to_string())?;

    let (amz_date, date) = amz_timestamp();
    let payload_hash = hex(&Sha256::digest(&payload));

    let mut signed_headers = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("host".to_string(), host.clone()),
        ("x-amz-date".to_string(), amz_date.clone()),
    ];
    if let Some(tok) = &creds.session_token {
        signed_headers.push(("x-amz-security-token".to_string(), tok.clone()));
    }
    signed_headers.sort();
    let header_names: Vec<&str> = signed_headers.iter().map(|(k, _)| k.as_str()).collect();
    let signed_header_list = header_names.join(";");
    let canonical_headers: String = signed_headers.iter().map(|(k, v)| format!("{k}:{}\n", v.trim())).collect();

    let canonical_request = format!("POST\n{canonical_path}\n\n{canonical_headers}\n{signed_header_list}\n{payload_hash}");
    let scope = format!("{date}/{region}/bedrock/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex(&Sha256::digest(canonical_request.as_bytes()))
    );

    let k_date = hmac_sha256(format!("AWS4{}", creds.secret_key).as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"bedrock");
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex(&hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_header_list}, Signature={signature}",
        creds.access_key
    );

    let mut req = ureq::post(&format!("https://{host}{path}"))
        .set("Content-Type", "application/json")
        .set("X-Amz-Date", &amz_date)
        .set("Authorization", &authorization)
        .timeout(Duration::from_secs(120));
    if let Some(tok) = &creds.session_token {
        req = req.set("X-Amz-Security-Token", tok);
    }
    let resp = req.send_bytes(&payload).map_err(|e| match e {
        ureq::Error::Status(code, r) => {
            let body = r.into_string().unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| v["message"].as_str().map(String::from))
                .unwrap_or_else(|| body.chars().take(300).collect());
            format!("bedrock HTTP {code}: {msg}")
        }
        ureq::Error::Transport(t) => format!("bedrock transport: {t}"),
    })?;
    let v: Value = resp.into_json().map_err(|e| e.to_string())?;
    let mut out = String::new();
    for block in v.pointer("/output/message/content").and_then(|c| c.as_array()).unwrap_or(&Vec::new()) {
        if let Some(text) = block["text"].as_str() {
            out.push_str(text);
        }
    }
    if out.is_empty() {
        return Err(format!("bedrock: empty response ({})", v["stopReason"].as_str().unwrap_or("?")));
    }
    Ok(out)
}

// --- small primitives, no chrono ------------------------------------------

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// ("YYYYMMDDTHHMMSSZ", "YYYYMMDD") in UTC, via the civil-from-days trick.
fn amz_timestamp() -> (String, String) {
    let secs = now_secs();
    let days = (secs / 86400) as i64;
    let (y, m, d) = civil_from_days(days);
    let rem = secs % 86400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let date = format!("{y:04}{m:02}{d:02}");
    (format!("{date}T{hh:02}{mm:02}{ss:02}Z"), date)
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// RFC3986 encode a path segment, keeping the model id readable in errors.
fn uri_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn parse_iso8601(s: &str) -> Option<u64> {
    // 2026-07-18T22:41:00Z (or +00:00); good enough for an expiry check.
    let s = s.trim_end_matches('Z');
    let (date, time) = s.split_once('T')?;
    let mut dp = date.split('-');
    let (y, m, d): (i64, u32, u32) = (dp.next()?.parse().ok()?, dp.next()?.parse().ok()?, dp.next()?.parse().ok()?);
    let time = time.split(['+', '-']).next()?;
    let mut tp = time.split(':');
    let (hh, mm): (u64, u64) = (tp.next()?.parse().ok()?, tp.next()?.parse().ok()?);
    let ss: u64 = tp.next().and_then(|v| v.split('.').next()).and_then(|v| v.parse().ok()).unwrap_or(0);
    let days = days_from_civil(y, m, d);
    Some((days as u64) * 86400 + hh * 3600 + mm * 60 + ss)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_roundtrip() {
        // 2026-07-18 is day 20652 since the epoch.
        assert_eq!(civil_from_days(20_652), (2026, 7, 18));
        assert_eq!(days_from_civil(2026, 7, 18), 20_652);
        assert_eq!(parse_iso8601("1970-01-01T00:00:10Z"), Some(10));
    }

    #[test]
    fn sigv4_primitives_match_aws_test_vector() {
        // The documented AWS example signing key vector.
        let k_date = hmac_sha256(b"AWS4wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", b"20120215");
        let k_region = hmac_sha256(&k_date, b"us-east-1");
        let k_service = hmac_sha256(&k_region, b"iam");
        let k_signing = hmac_sha256(&k_service, b"aws4_request");
        assert_eq!(
            hex(&k_signing),
            "f4780e2d9f65fa895f9c67b32ce1baf0b0d8a43505a000a1a9e090d414db404d"
        );
    }

    #[test]
    fn model_id_encoding() {
        assert_eq!(
            uri_encode("us.anthropic.claude-sonnet-5-20250929-v1:0"),
            "us.anthropic.claude-sonnet-5-20250929-v1%3A0"
        );
    }
}
