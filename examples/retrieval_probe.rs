//! Deterministic retrieval-composition probe (issues #14/#15/#24).
//!
//! The judge-graded end-to-end run is dominated by model + judge variance at
//! n=1: the same retrieval set yields PASS or FAIL run to run. But the graph's
//! actual claim is about *what gets retrieved*, which is deterministic. This
//! probe cuts out the model entirely and asks, for the crux queries: what does
//! the lexical route load, what does the entity route load, what does the
//! entity route with one co-occurrence hop load, and — the questions that
//! matter — is the storage-tier 140 present in the DRIVE query (the precision
//! bleed), and is the platform-team fact present in the ENG query (the reach)?
//!
//! Run: `cargo run --release --example retrieval_probe` (needs Ollama up with
//! nomic-embed-text). It ingests the exact disjoint transcript.

use continuum::driver::MemoryIndexDriver;
use continuum::hierarchical::HierarchicalTopicDriver;
use continuum::ollama::Ollama;

// (label, fact_a, fact_b) — the same eight disjoint cases the harness plants.
const FACTS: &[(&str, &str, &str)] = &[
    ("api", "My API plan allows 50 thousand requests per month.", "I've burned through about 62 thousand calls already."),
    ("flat", "The new flat is 2200 a month.", "My take home pay works out to 2600."),
    ("drive", "The external drive holds 500 gigabytes.", "My photo library weighs in at 620 gigabytes."),
    ("grant", "The grant deadline is March 10th.", "I get back from Tokyo on March 8th."),
    ("eng", "We budgeted for 12 engineers this year.", "There are 15 people on the platform team now."),
    ("laptop", "The flight to Berlin is 9 hours.", "This laptop runs about 6 hours on a charge."),
    ("gift", "I set aside 300 for presents this year.", "The wedding one cost 180 and the birthday one 140."),
    ("storage", "The basic tier caps at 100 gigabytes.", "I'm currently keeping 140 gigabytes up there."),
];
const DISTRACTORS: &[&str] = &[
    "What's a good warmup before a run?", "Tell me something about the moon.",
    "How do noise cancelling headphones work?", "I saw a heron by the river today.",
    "Explain eventual consistency without jargon.", "What makes sourdough different from regular bread?",
    "The gym was packed today.", "Recommend a novel for a long flight.",
];

// (query, the label whose facts we WANT reached, the label whose fact must NOT
// bleed in — "" if none).
const QUERIES: &[(&str, &str, &str)] = &[
    ("will the backup fit on the drive?", "drive", "storage"),
    ("are we over the engineering hiring budget?", "eng", ""),
    ("do I need to move off the basic tier?", "storage", ""),
    ("will my battery cover the entire Berlin flight?", "laptop", ""),
];

fn main() {
    let ol = Ollama::new("llama3.1:8b", "nomic-embed-text");
    let mut d = HierarchicalTopicDriver::new("/probe");
    d.set_embedder(ol.clone());

    // Ingest interleaved exactly like the harness: all A facts, distractors,
    // all B facts, distractors. Record which idx belongs to which label/slot.
    let mut label_of: Vec<(usize, String)> = Vec::new();
    let mut ts = 1000u64;
    for (lab, a, _) in FACTS { let i = d.ingest_turn_impl("user", a, &ts.to_string()); label_of.push((i, format!("{lab}.a"))); ts += 60; }
    for q in &DISTRACTORS[..4] { d.ingest_turn_impl("user", q, &ts.to_string()); ts += 60; }
    for (lab, _, b) in FACTS { let i = d.ingest_turn_impl("user", b, &ts.to_string()); label_of.push((i, format!("{lab}.b"))); ts += 60; }
    for q in &DISTRACTORS[4..] { d.ingest_turn_impl("user", q, &ts.to_string()); ts += 60; }

    let name = |idx: usize| -> String {
        label_of.iter().find(|(i, _)| *i == idx).map(|(_, l)| l.clone()).unwrap_or_else(|| format!("d{idx}"))
    };
    // Real facts are "label.a"/"label.b" (contain a dot); distractors are
    // "d{idx}" (no dot). Filter on the dot, not the leading letter — "drive"
    // starts with 'd' too.
    let labels = |idxs: &[usize]| -> Vec<String> {
        let mut v: Vec<String> = idxs.iter().map(|&i| name(i)).filter(|l| l.contains('.')).collect();
        v.sort();
        v
    };

    // Debug: for the drive query, dump the top entity-node scores so we can
    // see whether the "drive" node resolves as it must (cos ~1.0) or not.
    {
        let q = "will the backup fit on the drive?";
        let (ents, scores) = d.debug_entity_scores(q, 200);
        println!("DEBUG drive query entities: {ents:?}");
        // The decisive comparison: how close is the storage-tier distractor
        // ("keeping" -> the 140) versus the drive's OWN second fact
        // ("library"/"photo"/"weighs" -> the 620 it needs)?
        for target in ["drive", "keeping", "library", "photo", "weighs", "external", "holds"] {
            if let Some((_, s)) = scores.iter().find(|(n, _)| n == target) {
                let rank = scores.iter().position(|(n, _)| n == target).unwrap();
                println!("   {s:.3}  (rank {rank:>2})  {target}");
            }
        }
    }

    for (q, want, avoid) in QUERIES {
        let emb = ol.embed(q).unwrap_or_default();
        let lex = d.route_query(q, &emb);
        let ent = d.entity_route(q, false);
        let edg = d.entity_route(q, true);
        println!("\nQ: {q}");
        println!("  want reached: {want}.a/{want}.b   must NOT bleed: {}", if avoid.is_empty() { "-" } else { avoid });
        for (tag, idxs) in [("lexical    ", &lex), ("entity     ", &ent), ("entity+edge", &edg)] {
            let ls = labels(idxs);
            let reached = ls.iter().any(|l| l.starts_with(want));
            let reached_both = ls.contains(&format!("{want}.a")) && ls.contains(&format!("{want}.b"));
            let bled = !avoid.is_empty() && ls.iter().any(|l| l.starts_with(avoid));
            println!("    {tag} n={:2} reach={} both={} bleed={:5}  facts={:?}",
                idxs.len(), if reached {"Y"} else {"N"}, if reached_both {"Y"} else {"N"},
                if bled {"BLEED"} else {"ok"}, ls);
        }
    }
}
