"""Retrieval and answer evals: hit@3, MRR and refusal on unanswerable questions."""
import sys
from pathlib import Path
import yaml
from rag.answer import answer
from rag.index import Index

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    idx = Index(); idx.build(ROOT / "docs")
    cases = yaml.safe_load(open(ROOT / "evals/questions.yaml"))
    hits = rr = refusals_ok = n_ans = n_unans = 0
    for c in cases:
        if c["source"] is None:
            n_unans += 1
            refusals_ok += not answer(idx, c["q"]).grounded
            continue
        n_ans += 1
        ranked = [ch.source for ch, _ in idx.search(c["q"], 3)]
        if c["source"] in ranked:
            hits += 1; rr += 1 / (ranked.index(c["source"]) + 1)
    print(f"hit@3 {hits}/{n_ans} | MRR {rr / n_ans:.2f} | correct refusals {refusals_ok}/{n_unans}")
    return 0 if hits / n_ans >= 0.85 and refusals_ok == n_unans else 1


if __name__ == "__main__":
    sys.exit(main())
