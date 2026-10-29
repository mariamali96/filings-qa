"""Run every labelled question through the agent and write a report.

Outputs (in eval/results/):
  results_<timestamp>.json   every question, answer, citation and check
  report_<timestamp>.md      the metrics table, with Wilson 95% intervals
  review_<timestamp>.md      text answers to grade by hand (then re-run with grades)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from eval.report import review_sheet, summarize, to_markdown
from eval.scoring import score


def load_questions(paths: list[str]) -> list[dict]:
    questions: list[dict] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"(skipping missing {p})")
            continue
        questions += yaml.safe_load(path.read_text()).get("questions", [])
    return questions


def load_grades(path: Path = Path("eval/human_grades.yaml")) -> dict[str, str]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def run(agent, questions: list[dict]) -> list[dict]:
    results = []
    for q in questions:
        answer, _evidence, _plan = agent.ask(q["question"])
        results.append(score(q, answer))
        print(f"{q['id']}: correct={results[-1].get('correct')} issues={len(answer.issues)}")
    return results


def main(question_files: list[str], out_dir: str) -> None:  # pragma: no cover - network
    from filings_qa.runtime import build_agent

    questions = load_questions(question_files)
    results = run(build_agent(), questions)
    summary = summarize(results, load_grades())

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{stamp}.json").write_text(json.dumps(results, indent=2))
    (out / f"report_{stamp}.md").write_text(to_markdown(summary))
    (out / f"review_{stamp}.md").write_text(review_sheet(results))
    print()
    print(to_markdown(summary))
