//! LongMemEval runner (third-party benchmark, per-category).
//!
//! LoCoMo reports one aggregate, which averages away exactly the capability our
//! own harnesses keep failing: composition across sessions. LongMemEval splits
//! its 500 questions into six categories (single-session user / assistant /
//! preference, multi-session, temporal-reasoning, knowledge-update), so it can
//! confirm or refute the shape our harnesses found, on someone else's data.
//!
//! Deliberately mirrors `eval.rs`: same driver, same `Kernel`, same
//! `SYSTEM_TEMPLATE`, same 30-message cap, same single-pass retrieval plus one
//! fault. The only thing that varies is the answer model.
//!
//! Each question carries its OWN haystack (~50 sessions, ~490 turns), so every
//! question gets a fresh driver, ingests its haystack, then asks once. Output is
//! one jsonl per category, which lets `bench/judge_frontier.py` grade them
//! unchanged and report per-category (it already prints per-file).
//!
//! Usage: longmemeval [--data PATH] [--n-per-cat N] [--model M] [--out-dir DIR]

use std::collections::BTreeMap;
use std::io::Write;

use continuum::driver::MemoryIndexDriver;
use continuum::hierarchical::HierarchicalTopicDriver;
use continuum::kernel::{Kernel, KernelConfig};
use continuum::ollama::{ChatMessage, Ollama};

/// The daemon's calculator rule, verbatim, appended to the kernel's
/// SYSTEM_TEMPLATE to form a SECOND, distinct prompt. SYSTEM_TEMPLATE is not
/// edited: the point is to measure the delta between the two paths, so they
/// have to stay separate.
const CALC_RULE: &str = "\n- NEVER do arithmetic on remembered numbers or dates in your head (sums, \
differences, comparisons against limits, date shifts). Respond with EXACTLY\n  \
CALC_NEEDED: <expression>\n  and nothing else, e.g. `CALC_NEEDED: 1800 + 200` \
or `CALC_NEEDED: October 14 + 7 days`. The exact result comes back to you.\n";

/// Pull `PREFIX: rest` out of a reply the way the daemon's protocol parser does.
fn protocol_request(reply: &str, prefix: &str) -> Option<String> {
    reply.lines().find_map(|l| {
        let t = l.trim();
        t.find(prefix).map(|p| t[p + prefix.len()..].trim().to_string())
    }).filter(|s| !s.is_empty())
}

use continuum::hierarchical::normalize_benchmark_date as normalize_date;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut data = "data/longmemeval/longmemeval_s.json".to_string();
    let mut n_per_cat: usize = 20;
    let mut model = "llama3.1:8b".to_string();
    let mut out_dir = "fullbench".to_string();
    let mut tag = String::new();
    // Structural pass: ingest + route only, no generation. The routed set is
    // deterministic (embeddings + BM25), so reachability measured here is valid
    // for the graded run over the same sample.
    let mut structural_only = false;
    let mut ungate = false;
    let mut annotate = false;
    // Same plumbing as eval: only the completion call moves. Retrieval,
    // embeddings, prompts and the cap are unchanged.
    let mut calc_path = false;
    let mut provider = "ollama".to_string();
    let mut region: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { data = args[i + 1].clone(); i += 2; }
            "--n-per-cat" => { n_per_cat = args[i + 1].parse().unwrap_or(n_per_cat); i += 2; }
            "--model" => { model = args[i + 1].clone(); i += 2; }
            "--out-dir" => { out_dir = args[i + 1].clone(); i += 2; }
            "--tag" => { tag = args[i + 1].clone(); i += 2; }
            "--structural-only" => { structural_only = true; i += 1; }
            "--ungate" => { ungate = true; i += 1; }
            "--annotate" => { annotate = true; i += 1; }
            "--calc" => { calc_path = true; i += 1; }
            "--provider" => { provider = args[i + 1].clone(); i += 2; }
            "--region" => { region = Some(args[i + 1].clone()); i += 2; }
            other => { eprintln!("unknown arg {other}"); i += 1; }
        }
    }
    if tag.is_empty() {
        tag = model.replace([':', '/', '.'], "_");
    }
    std::fs::create_dir_all(&out_dir).ok();

    let ollama = Ollama::new(&model, "nomic-embed-text");
    eprintln!("[loading {data} ...]");
    let raw = std::fs::read_to_string(&data).expect("read dataset");
    let all: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("parse dataset");
    eprintln!("[{} questions loaded]", all.len());

    // Deterministic stratified sample: group by category, sort by question_id,
    // take the first n. Same subset every run, so tiers are comparable.
    let mut by_cat: BTreeMap<String, Vec<&serde_json::Value>> = BTreeMap::new();
    for q in &all {
        let cat = q.get("question_type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
        by_cat.entry(cat).or_default().push(q);
    }
    for v in by_cat.values_mut() {
        v.sort_by_key(|q| q.get("question_id").and_then(|x| x.as_str()).unwrap_or("").to_string());
        v.truncate(n_per_cat);
    }
    let total: usize = by_cat.values().map(|v| v.len()).sum();
    eprintln!("[sampled {total} questions across {} categories, n<={n_per_cat} each]", by_cat.len());

    let t_start = std::time::Instant::now();
    let mut done = 0usize;
    for (cat, qs) in &by_cat {
        let path = format!("{out_dir}/lme_{}_{}.jsonl", cat.replace('-', "_"), tag);
        let mut f = std::fs::File::create(&path).expect("create jsonl");
        for q in qs {
            let question = q.get("question").and_then(|v| v.as_str()).unwrap_or("");
            // 32 of the 500 golds are JSON numbers, not strings (mostly
            // multi-session counting questions). as_str() would silently yield
            // an empty gold and make them unjudgeable, so render either form.
            let gold = match q.get("answer") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            let gold = gold.as_str();
            let qid = q.get("question_id").and_then(|v| v.as_str()).unwrap_or("");

            // Fresh memory per question: the haystack belongs to this question.
            let mut d = HierarchicalTopicDriver::new("/social");
            d.set_embedder(ollama.clone());
            let sessions = q.get("haystack_sessions").and_then(|v| v.as_array());
            let dates = q.get("haystack_dates").and_then(|v| v.as_array());
            let mut turns = 0usize;
            // (message idx, session idx) for every turn LongMemEval marks as
            // carrying the answer.
            let mut evidence: Vec<(usize, usize)> = Vec::new();
            if let Some(sessions) = sessions {
                for (si, sess) in sessions.iter().enumerate() {
                    let ts = dates
                        .and_then(|d| d.get(si))
                        .and_then(|v| v.as_str())
                        .map(normalize_date)
                        .unwrap_or_default();
                    if let Some(list) = sess.as_array() {
                        for t in list {
                            let role = t.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                            let text = t.get("content").and_then(|v| v.as_str()).unwrap_or("");
                            if !text.is_empty() {
                                let idx = d.ingest_turn(role, text, &ts);
                                if t.get("has_answer").and_then(|v| v.as_bool()).unwrap_or(false) {
                                    evidence.push((idx, si));
                                }
                                turns += 1;
                            }
                        }
                    }
                }
            }

            // Structural instrumentation, separate from capability. LongMemEval
            // haystacks are far larger than LoCoMo's and retrieval is capped at
            // 30 messages, so a multi-session question can be unanswerable for
            // budget reasons rather than synthesis reasons. Recompute the same
            // route page_in will take (same embedding, same route_query) purely
            // to see which evidence turns could have reached the model. This
            // does not change what the model is shown.
            // Flags must be set BEFORE the route is computed: the structural
            // pass measures this exact route, so setting them later silently
            // measured the baseline and made --ungate look like a no-op.
            d.route_cfg.ungate_dense = ungate;
            d.route_cfg.annotate_values = annotate;
            let q_emb = ollama.embed(question).unwrap_or_default();
            let routed = d.route_query(question, &q_emb);
            // Capture the driver's routing trace so --ungate is auditable in
            // the output rather than assumed to have taken effect.
            let route_trace = d.last_path.borrow().clone();
            let ev_total = evidence.len();
            let ev_loaded = evidence.iter().filter(|(i, _)| routed.contains(i)).count();
            let ev_sessions_total: std::collections::BTreeSet<usize> =
                evidence.iter().map(|(_, s)| *s).collect();
            let ev_sessions_loaded: std::collections::BTreeSet<usize> = evidence
                .iter()
                .filter(|(i, _)| routed.contains(i))
                .map(|(_, s)| *s)
                .collect();

            let (pred, faulted, loaded) = if structural_only {
                (String::new(), false, routed.len())
            } else {
                let mut kernel = Kernel::new(ollama.clone(), KernelConfig::default());
                kernel.mount(Box::new(d));
                if provider == "bedrock" {
                    let region = region.clone().unwrap_or_else(continuum::bedrock::default_region);
                    let model_id = model.clone();
                    kernel.set_chat_override(Box::new(move |messages, max_tokens| {
                        let (system, turns) = continuum::bedrock::converse_messages(messages);
                        continuum::bedrock::converse(&region, &model_id, &system, &turns, max_tokens, 0.0)
                    }));
                }
                if calc_path {
                    // Daemon-equivalent path: same retrieval, CALC-enabled prompt,
                    // and the same action loop (raise CALC_NEEDED, evaluate
                    // deterministically, feed the exact value back, re-answer).
                    let template = format!("{}{}", continuum::kernel::SYSTEM_TEMPLATE, CALC_RULE);
                    let (mut messages, meta) = kernel.prepare_with(question, &[], &template);
                    let mut reply = kernel.complete_messages(&messages).unwrap_or_default();
                    let mut rounds = 0;
                    let mut seen: Vec<String> = Vec::new();
                    while rounds < 3 {
                        let Some(expr) = protocol_request(&reply, "CALC_NEEDED:") else { break };
                        if seen.contains(&expr) { break; }
                        seen.push(expr.clone());
                        let feedback = match continuum::calc::eval(&expr) {
                            Ok(v) => format!("[CALC RESULT] {expr} = {v}\nUse this exact value in your answer."),
                            Err(e) => format!("[CALC ERROR] {e}\nState the calculation in words instead of guessing a number."),
                        };
                        messages.push(ChatMessage::new("assistant", reply.clone()));
                        messages.push(ChatMessage::new("user", feedback));
                        reply = kernel.complete_messages(&messages).unwrap_or_default();
                        rounds += 1;
                    }
                    (reply.trim().to_string(), meta.page_faulted, meta.messages_loaded)
                } else {
                    let r = kernel.query(question, &[]);
                    (r.response.trim().to_string(), r.page_faulted, r.messages_loaded)
                }
            };

            let rec = serde_json::json!({
                "qid": qid, "cat": cat, "question": question, "gold": gold, "pred": pred,
                "fault": faulted, "loaded": loaded,
                "haystack_turns": turns, "model": model, "route_trace": route_trace, "path": if calc_path {"calc"} else {"system_template"},
                "evidence_turns_total": ev_total, "evidence_turns_loaded": ev_loaded,
                "evidence_sessions_total": ev_sessions_total.len(),
                "evidence_sessions_loaded": ev_sessions_loaded.len(),
                "routed_total": routed.len(),
            });
            writeln!(f, "{rec}").ok();
            done += 1;
            eprintln!(
                "[{done}/{total}] {cat} {qid} turns={turns} loaded={loaded} ev={ev_loaded}/{ev_total} sess={}/{} ({:.0}s)",
                ev_sessions_loaded.len(), ev_sessions_total.len(), t_start.elapsed().as_secs_f32()
            );
        }
        eprintln!("[wrote {path}]");
    }
    eprintln!(
        "[done: {done} questions in {:.1} min] grade with:\n  python3 bench/judge_frontier.py \"{out_dir}/lme_*_{tag}.jsonl\"",
        t_start.elapsed().as_secs_f32() / 60.0
    );
}
