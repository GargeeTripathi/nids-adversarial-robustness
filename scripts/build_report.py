"""
Assemble the three documentation parts into one continuous report.

Kept as a build step rather than maintaining a single giant file by hand: the parts are
edited independently as experiment phases land, and concatenating on demand avoids the
usual drift between "the source of truth" and "the thing people actually read".

    .venv/Scripts/python.exe scripts/build_report.py            # markdown
    .venv/Scripts/python.exe scripts/build_report.py --html     # standalone HTML too
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

PARTS = [
    "part1_foundations.md",
    "part2_methodology.md",
    "part3_results.md",
]

TITLE = "Adversarial Robustness & Cross-Domain Transferability of Deep-Learning NIDS"

FRONT_MATTER = f"""# {TITLE}

*A study of how deep-learning network intrusion detectors fail under adversarial pressure,
and whether those failures are real or artefacts of an unrealistic threat model.*

**Generated:** {date.today().isoformat()}
**Code:** `D:\\NIDS\\src\\nids\\` — **Results:** `D:\\NIDS\\outputs\\`

---

## Executive summary

Deep-learning intrusion detectors are widely reported to collapse under adversarial attack,
with published evasion rates approaching 100%. This study reproduces those numbers and then
asks a question the standard methodology skips: **could the adversarial traffic actually be
sent?**

For flow-based NIDS the answer is usually no. A gradient attack run with the conventional
image-derived threat model proposes flows with negative durations, fractional packet counts,
protocols that are 30% TCP and 70% UDP, and byte counts the *victim* supposedly sent
decreasing on the attacker's command. Measured here: **100% of unconstrained adversarial
flows violate at least one physical constraint.**

Re-running every attack under a projection onto the set of physically realizable flows
changes the conclusions substantially:

| Finding | Conventional evaluation | Realizable-traffic evaluation |
|---|---|---|
| Peak reported evasion success | ~100% at every budget | 12%-96% depending on architecture |
| Overstatement factor | — | up to **17.5x** at small budgets |
| Black-box transfer between architectures | 28.9% mean | **4.4%** mean |
| Apparent effect of architecture choice | negligible (all saturate) | **8x** difference in vulnerability |

Four results follow, each established on real runs and reported with its caveats:

1. **Unconstrained adversarial evaluation overstates attack success by up to 17.5x**, and
   the overstatement is largest at small perturbation budgets - precisely the regime where a
   stealthy attack would operate.
2. **Architecture changes adversarial exposure by roughly 8x while changing clean accuracy
   by under 4 points.** The CNN evaluated here has the best false-positive rate of the four
   and is by far the most fragile. Clean-accuracy benchmarks carry no information about
   robustness.
3. **FGSM understates vulnerability in three of four architectures**, by up to 2.9x. A
   robustness claim supported only by a single-step attack is not supported.
4. **Adversarial examples transfer poorly between architectures on tabular flow data**
   (4.4% under realistic constraints), contradicting the image-domain expectation.
   Architectural diversity is a cheap and genuine defence in this setting.

A fifth result concerns evaluation protocol rather than adversarial robustness: on NSL-KDD's
**official** train/test split, which deliberately withholds entire attack families from
training, every model loses roughly 19-23 accuracy points relative to an i.i.d. validation
split. Results near 99% - common in the literature - require pooling and re-splitting the
official files, which converts a zero-day generalisation test into an interpolation test.

**What is not established.** Cross-domain transferability (training on one dataset and
testing on another) requires UNSW-NB15 and CICIDS2017, both gated behind registration forms.
The relevant sections state the structural limit that is already measurable - the genuine
three-way shared feature space is five columns - and are otherwise left empty rather than
filled with plausible values.

---

## Contents

- **Part I — Foundations.** Intrusion detection from first principles; adversarial machine
  learning; the three datasets; why network traffic is not an image; glossary.
- **Part II — Methodology.** Experimental design; the four architectures and their
  limitations; preprocessing; attack implementations; software.
- **Part III — Results and Analysis.** Baselines, evasion, transfer, poisoning, defence,
  cross-domain; threats to validity; conclusions.

---

"""


def demote_headings(text: str) -> str:
    """
    Push every heading down one level so the parts nest under the report title.

    Applied only to real headings - a `#` inside a fenced code block is a comment, not a
    heading, so fences are tracked and their contents left alone.
    """
    out, in_fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and re.match(r"^#{1,5}\s", line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def build_markdown() -> str:
    chunks = [FRONT_MATTER]
    for name in PARTS:
        path = DOCS_DIR / name
        if not path.exists():
            print(f"  WARNING: missing {name}")
            continue
        chunks.append(demote_headings(path.read_text(encoding="utf-8")))
        chunks.append("\n\n---\n\n")
        print(f"  included {name}")
    return "\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", action="store_true", help="also emit standalone HTML")
    args = parser.parse_args()

    print("Assembling report")
    markdown = build_markdown()

    out_md = DOCS_DIR / "NIDS_Adversarial_Robustness_Report.md"
    out_md.write_text(markdown, encoding="utf-8")

    words = len(markdown.split())
    print(f"\nWrote {out_md}")
    print(f"  {len(markdown):,} characters, ~{words:,} words")

    if args.html:
        try:
            import markdown as md_lib
        except ImportError:
            print("  (skipping HTML: `pip install markdown` first)")
            return
        body = md_lib.markdown(markdown, extensions=["tables", "fenced_code", "toc"])
        out_html = DOCS_DIR / "NIDS_Adversarial_Robustness_Report.html"
        out_html.write_text(body, encoding="utf-8")
        print(f"  Wrote {out_html}")


if __name__ == "__main__":
    main()
