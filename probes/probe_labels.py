"""Shared labeling predicate for the Python retrieval probes — mirror of the
Rust src/probe_util.rs. Planted facts are "<case>.a"/"<case>.b"; distractors
are "d0".."d7". Filter on the dot, NEVER on a leading 'd' ("drive" starts with
'd' and got silently dropped twice). Import keep_fact() everywhere.

Run `python probe_labels.py` to execute the regression self-test."""


def keep_fact(label: str) -> bool:
    """True for a planted fact label (has a dot), False for a distractor."""
    return "." in label


if __name__ == "__main__":
    for case in ["drive", "storage", "eng", "laptop", "api", "flat", "grant", "gift"]:
        assert keep_fact(f"{case}.a"), f"{case}.a dropped"
        assert keep_fact(f"{case}.b"), f"{case}.b dropped"
    for i in range(8):
        assert not keep_fact(f"d{i}"), f"d{i} kept"
    print("probe_labels regression self-test: PASS (drive.a/drive.b survive)")
