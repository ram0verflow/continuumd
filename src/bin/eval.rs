//! ROUGE evaluation of the AIOS kernel on the LoCoMo benchmark.
//!
//! For each QA pair in a LoCoMo conversation we:
//!   1. route the question through the `HierarchicalTopicDriver`,
//!   2. let the kernel assemble a namespaced context + run the page-fault loop,
//!   3. score the model's answer against the gold answer with ROUGE-1 / ROUGE-L.
//!
//! Usage: eval [--limit N] [--conv I] [--model M] [--skip-adversarial]
//!
//! LoCoMo category 5 = adversarial (the answer is *not* in the conversation).
//! For those we don't want ROUGE overlap, we want a page fault / refusal. We
//! report that separately as "adversarial handled".

use std::time::Instant;

use aios::driver::MemoryIndexDriver;
use aios::hierarchical::HierarchicalTopicDriver;
use aios::kernel::{detect_page_fault, Kernel, KernelConfig};
use aios::ollama::{ChatMessage, Ollama};

/// LLM-as-judge: does the prediction convey the gold answer? The field norm
/// for LoCoMo (raw ROUGE against terse golds punishes correct sentence-length
/// answers). Same local model, temp 0, YES/NO.
fn judge(ollama: &Ollama, question: &str, gold: &str, pred: &str) -> Option<bool> {
    let prompt = format!(
        "Question: {question}\nGold answer: {gold}\nModel answer: {pred}\n\n\
         Does the model answer convey the same key information as the gold answer? \
         Minor wording/format differences are fine. Reply with exactly one word: YES or NO."
    );
    let msgs = [
        ChatMessage::new("system", "You are a strict but fair grader. Reply only YES or NO."),
        ChatMessage::new("user", prompt),
    ];
    let resp = ollama.chat(&msgs, 2048, 5).ok()?;
    let up = resp.to_uppercase();
    Some(up.contains("YES"))
}

fn main() {
    let mut limit = 40usize;
    let mut skip = 0usize;
    let mut only_cat: Option<String> = None;
    let mut jsonl_path: Option<String> = None;
    let mut conv_idx = 0usize;
    let mut model = "llama3.1:8b".to_string();
    let mut skip_adversarial = false;
    let mut use_judge = true;
    let mut adv_only = false;
    let mut judge_model: Option<String> = None;
    let mut ablate: Vec<String> = Vec::new();

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => { limit = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(limit); i += 2; }
            "--skip" => { skip = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "--only-cat" => { only_cat = args.get(i + 1).cloned(); i += 2; }
            "--jsonl" => { jsonl_path = args.get(i + 1).cloned(); i += 2; }
            "--conv" => { conv_idx = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(conv_idx); i += 2; }
            "--model" => { model = args.get(i + 1).cloned().unwrap_or(model); i += 2; }
            "--judge-model" => { judge_model = args.get(i + 1).cloned(); i += 2; }
            "--skip-adversarial" => { skip_adversarial = true; i += 1; }
            "--no-judge" => { use_judge = false; i += 1; }
            "--adv-only" => { adv_only = true; i += 1; }
            "--ablate" => {
                ablate = args.get(i + 1).map(|s| s.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
                i += 2;
            }
            _ => i += 1,
        }
    }

    let locomo_path = "data/locomo10.json";
    let tree_path = format!("data/conv_{conv_idx}.json");

    let locomo: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(locomo_path).expect("read locomo")).expect("json");
    let conv = &locomo.as_array().expect("array")[conv_idx];
    let sample_id = conv.get("sample_id").and_then(|v| v.as_str()).unwrap_or("?");
    let qa = conv.get("qa").and_then(|v| v.as_array()).expect("qa");

    let ollama = Ollama::new(&model, "nomic-embed-text");
    if !ollama.healthy() {
        eprintln!("ERROR: Ollama not reachable / models not loaded. Is `ollama serve` running?");
        std::process::exit(1);
    }

    eprintln!("=== AIOS ROUGE eval ===");
    eprintln!("conv {conv_idx} ({sample_id}), model {model}, {} QA pairs available", qa.len());
    let t0 = Instant::now();
    // Conv 0 has a prebuilt (Claude-partitioned) tree; other convs are built
    // via online ingestion, the stress test showed the online-grown tree
    // routes as well as the prebuilt one (74.0% vs 72.1%).
    let mut driver = match std::fs::read_to_string(&tree_path) {
        Ok(s) => {
            eprintln!("[prebuilt tree: building name embeddings…]");
            let tree_data: serde_json::Value = serde_json::from_str(&s).expect("tree json");
            HierarchicalTopicDriver::from_conv_json("/social", &tree_data, Some(&ollama))
        }
        Err(_) => {
            eprintln!("[no prebuilt tree: online ingestion of conversation…]");
            let mut d = HierarchicalTopicDriver::new("/social");
            d.set_embedder(ollama.clone());
            for (speaker, text, ts) in conv_turns(conv) {
                d.ingest_turn(&speaker, &text, &ts);
            }
            d
        }
    };
    for a in &ablate {
        match a.as_str() {
            "tree" => driver.route_cfg.use_tree = false,
            "bm25" => driver.route_cfg.use_bm25 = false,
            "dense" => driver.route_cfg.use_dense = false,
            "cap" => driver.route_cfg.max_load = 1000,
            "resolver" => driver.route_cfg.temporal_notes = false,
            other => eprintln!("unknown ablation: {other}"),
        }
    }
    if !ablate.is_empty() {
        eprintln!("[ablated: {}]", ablate.join(","));
    }
    eprintln!(
        "[driver: {} messages, {} leaves, depth {}, embeddings built in {:.1}s]",
        driver.message_len(),
        driver.tree().map(|t| t.leaf_count()).unwrap_or(0),
        driver.tree().map(|t| t.depth()).unwrap_or(0),
        t0.elapsed().as_secs_f32()
    );

    // Judge with a fixed model (default: answer model) so tuned-vs-baseline
    // comparisons share the same grader.
    let judge_handle = match &judge_model {
        Some(jm) => Ollama::new(jm, "nomic-embed-text"),
        None => ollama.clone(),
    };
    let mut kernel = Kernel::new(ollama, KernelConfig::default());
    kernel.mount(Box::new(driver));

    // Accumulators
    let mut n = 0usize;
    let mut sum_r1 = 0.0f64;
    let mut sum_rl = 0.0f64;
    let mut hits = 0usize; // gold substring appears in answer
    let mut faults = 0usize;
    let mut judged = 0usize; // non-adversarial judged
    let mut judge_correct = 0usize;
    let mut cat_scores: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
    let mut cat_judge: std::collections::BTreeMap<String, (usize, usize)> = Default::default();

    // Adversarial (category 5) handling
    let mut adv_total = 0usize;
    let mut adv_handled = 0usize;

    // Systems metrics: memory-side vs generation-side time per query.
    let mut retr_ms: Vec<f64> = Vec::new();
    let mut gen_ms: Vec<f64> = Vec::new();

    let mut jsonl_file = jsonl_path.as_ref().map(|p| {
        std::fs::File::create(p).expect("create jsonl")
    });

    // --adv-only: evaluate ONLY the unanswerable adversarial rows, those with
    // an `adversarial_answer` (the trap) instead of `answer`. The correct
    // behavior is a page fault / refusal; the failure mode is taking the bait.
    if adv_only {
        let mut total = 0usize;
        let mut handled = 0usize;
        let mut trapped = 0usize;
        for q in qa.iter() {
            if q.get("answer").is_some() {
                continue;
            }
            let Some(trap) = q.get("adversarial_answer").and_then(|v| v.as_str()) else { continue };
            let question = q.get("question").and_then(|v| v.as_str()).unwrap_or("");
            total += 1;
            let result = kernel.query(question, &[]);
            let answer = result.response.trim().to_string();
            let low = answer.to_lowercase();
            let refused = result.page_faulted
                || detect_page_fault(&answer).is_some()
                || [
                    "no mention", "not mentioned", "does not mention", "doesn't mention",
                    "never mentioned", "no information", "there is no", "not specified",
                    "cannot find", "can't find", "not stated",
                ]
                .iter()
                .any(|p| low.contains(p));
            let baited = !trap.is_empty() && low.contains(&trap.to_lowercase());
            if refused && !baited {
                handled += 1;
            }
            if baited {
                trapped += 1;
            }
            if let Some(f) = jsonl_file.as_mut() {
                use std::io::Write;
                let rec = serde_json::json!({
                    "cat": "5u", "adv": true, "adv_handled": refused && !baited,
                    "trapped": baited, "question": question, "trap": trap, "pred": answer,
                });
                writeln!(f, "{rec}").ok();
            }
            eprintln!(
                "[{total:>3}] handled={} trapped={} | Q: {} | pred: {}",
                if refused && !baited { "Y" } else { "." },
                if baited { "Y" } else { "." },
                truncate(question, 48),
                truncate(&answer, 60)
            );
        }
        println!("\n========== ADVERSARIAL (unanswerable) ==========");
        println!("total {total} | refused/faulted correctly: {handled} ({:.1}%) | took the bait: {trapped}",
            100.0 * handled as f64 / total.max(1) as f64);
        println!("=================================================");
        return;
    }

    let run_start = Instant::now();
    // `qid` enumerates scoreable rows (those with an answer field) across the
    // WHOLE qa list, stable across runs, so --skip/--only-cat slices from
    // different runs merge cleanly. Matches the [n/199] numbering of full runs.
    let mut qid = 0usize;
    for q in qa.iter() {
        let question = q.get("question").and_then(|v| v.as_str()).unwrap_or("");
        let gold_raw = q.get("answer");
        let gold = match gold_raw {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => continue, // unanswerable adversarial rows; see --adv-only
        };
        qid += 1;
        if qid <= skip {
            continue;
        }
        if qid > limit {
            break;
        }
        let category = q
            .get("category")
            .map(|c| match c {
                serde_json::Value::Number(nn) => nn.to_string(),
                serde_json::Value::String(s) => s.clone(),
                _ => "?".into(),
            })
            .unwrap_or_else(|| "?".into());
        if let Some(oc) = &only_cat {
            if &category != oc {
                continue;
            }
        }
        let is_adversarial = category == "5";
        if is_adversarial && skip_adversarial {
            continue;
        }

        let result = kernel.query(question, &[]);
        retr_ms.push(result.retrieval_ms);
        gen_ms.push(result.generation_ms);
        let answer = result.response.trim().to_string();
        n += 1;

        let mut adv_ok = false;
        if is_adversarial {
            adv_total += 1;
            // "Handled" = model page-faulted or refused rather than fabricating.
            if result.page_faulted || detect_page_fault(&answer).is_some() {
                adv_handled += 1;
                adv_ok = true;
            }
        }

        let r1 = rouge_n(&answer, &gold, 1);
        let rl = rouge_l(&answer, &gold);
        sum_r1 += r1;
        sum_rl += rl;
        if answer.to_lowercase().contains(&gold.to_lowercase()) && !gold.is_empty() {
            hits += 1;
        }
        if result.page_faulted {
            faults += 1;
        }
        let entry = cat_scores.entry(category.clone()).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += rl;

        // LLM-judge (non-adversarial only; adversarial correctness = refusal).
        let mut jmark = "-";
        if use_judge && !is_adversarial {
            if let Some(ok) = judge(&judge_handle, question, &gold, &answer) {
                judged += 1;
                let cj = cat_judge.entry(category.clone()).or_insert((0, 0));
                cj.0 += 1;
                if ok {
                    judge_correct += 1;
                    cj.1 += 1;
                    jmark = "Y";
                } else {
                    jmark = "n";
                }
            }
        }

        if let Some(f) = jsonl_file.as_mut() {
            use std::io::Write;
            let rec = serde_json::json!({
                "qid": qid, "cat": category, "r1": r1, "rl": rl,
                "judge": jmark, "fault": result.page_faulted,
                "adv": is_adversarial, "adv_handled": adv_ok,
                "loaded": result.messages_loaded,
                "question": question, "gold": gold, "pred": answer,
            });
            writeln!(f, "{rec}").ok();
        }

        let tag = if is_adversarial { " [ADV]" } else { "" };
        eprintln!(
            "[{qid:>3}/{limit}] cat{category}{tag} R1={r1:.2} RL={rl:.2} J={jmark} loaded={:>3} fault={} | Q: {}",
            result.messages_loaded,
            if result.page_faulted { "Y" } else { "." },
            truncate(question, 58)
        );
        eprintln!("        gold: {:<40} | pred: {}", truncate(&gold, 40), truncate(&answer, 70));
    }

    let elapsed = run_start.elapsed().as_secs_f32();
    println!("\n================= AIOS ROUGE RESULTS =================");
    println!("conv {conv_idx} ({sample_id}) | model {model} | scored {n} QA in {elapsed:.0}s");
    println!("-----------------------------------------------------");
    println!("ROUGE-1 F1 (mean): {:.4}", sum_r1 / n.max(1) as f64);
    println!("ROUGE-L F1 (mean): {:.4}", sum_rl / n.max(1) as f64);
    println!("Exact-contains   : {}/{} ({:.1}%)", hits, n, 100.0 * hits as f64 / n.max(1) as f64);
    if judged > 0 {
        println!(
            "LLM-judge acc    : {}/{} ({:.1}%)",
            judge_correct,
            judged,
            100.0 * judge_correct as f64 / judged as f64
        );
    }
    println!("Page faults fired: {faults}/{n}");
    if adv_total > 0 {
        println!("Adversarial (cat5) handled (page-fault/refuse): {adv_handled}/{adv_total}");
    }
    retr_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gen_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |v: &Vec<f64>, p: usize| v.get(v.len() * p / 100).copied().unwrap_or(0.0);
    if !retr_ms.is_empty() {
        println!(
            "Latency ms (p50/p95): retrieval {:.0}/{:.0} | generation {:.0}/{:.0}",
            pct(&retr_ms, 50), pct(&retr_ms, 95), pct(&gen_ms, 50), pct(&gen_ms, 95)
        );
    }
    println!("-----------------------------------------------------");
    println!("ROUGE-L by category (and judge acc):");
    for (cat, (cnt, sum)) in &cat_scores {
        let j = cat_judge
            .get(cat)
            .map(|(jn, jc)| format!("  judge {jc}/{jn}"))
            .unwrap_or_default();
        println!("  cat {cat}: {:.4}  (n={cnt}){j}", sum / *cnt as f64);
    }
    println!("=====================================================");
}

/// Extract (speaker, text, session_timestamp) turns in order from one conv.
fn conv_turns(conv: &serde_json::Value) -> Vec<(String, String, String)> {
    let c = conv.get("conversation").and_then(|v| v.as_object()).expect("conversation");
    let mut session_nums: Vec<u32> = c
        .keys()
        .filter_map(|k| {
            k.strip_prefix("session_")
                .filter(|r| !r.contains("date"))
                .and_then(|r| r.parse().ok())
        })
        .collect();
    session_nums.sort_unstable();

    let mut out = Vec::new();
    for sn in session_nums {
        let ts = c
            .get(&format!("session_{sn}_date_time"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if let Some(turns) = c.get(&format!("session_{sn}")).and_then(|v| v.as_array()) {
            for t in turns {
                let speaker = t.get("speaker").and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
                let text = t.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                if !text.is_empty() {
                    out.push((speaker, text, ts.clone()));
                }
            }
        }
    }
    out
}

fn truncate(s: &str, n: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= n {
        s
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push('…');
        out
    }
}

fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// ROUGE-N F1 between a prediction and a reference (gold), N-gram overlap.
fn rouge_n(pred: &str, gold: &str, n: usize) -> f64 {
    let pt = ngrams(&tokenize(pred), n);
    let gt = ngrams(&tokenize(gold), n);
    if pt.is_empty() || gt.is_empty() {
        return 0.0;
    }
    // Multiset overlap
    let mut gcount: std::collections::HashMap<&Vec<String>, usize> = Default::default();
    for g in &gt {
        *gcount.entry(g).or_insert(0) += 1;
    }
    let mut overlap = 0usize;
    for p in &pt {
        if let Some(c) = gcount.get_mut(p) {
            if *c > 0 {
                *c -= 1;
                overlap += 1;
            }
        }
    }
    f1(overlap, pt.len(), gt.len())
}

fn ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if tokens.len() < n {
        return Vec::new();
    }
    (0..=tokens.len() - n).map(|i| tokens[i..i + n].to_vec()).collect()
}

/// ROUGE-L F1 based on longest common subsequence of tokens.
fn rouge_l(pred: &str, gold: &str) -> f64 {
    let p = tokenize(pred);
    let g = tokenize(gold);
    if p.is_empty() || g.is_empty() {
        return 0.0;
    }
    let lcs = lcs_len(&p, &g);
    f1(lcs, p.len(), g.len())
}

fn lcs_len(a: &[String], b: &[String]) -> usize {
    let mut dp = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        let mut prev = 0;
        for j in 1..=b.len() {
            let tmp = dp[j];
            if a[i - 1] == b[j - 1] {
                dp[j] = prev + 1;
            } else {
                dp[j] = dp[j].max(dp[j - 1]);
            }
            prev = tmp;
        }
    }
    dp[b.len()]
}

fn f1(overlap: usize, pred_len: usize, gold_len: usize) -> f64 {
    if overlap == 0 {
        return 0.0;
    }
    let precision = overlap as f64 / pred_len as f64;
    let recall = overlap as f64 / gold_len as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}
