"""Command line:

filings-qa ingest AAPL MSFT NVDA --years 2
filings-qa ask "What were Apple's total net sales in fiscal 2024?"
filings-qa build-gold            # numeric ground truth from XBRL
filings-qa eval                  # run the evaluation suite
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _cmd_ingest(args: argparse.Namespace) -> None:  # pragma: no cover - network
    from filings_qa.config import get_settings
    from filings_qa.edgar import EdgarClient
    from filings_qa.embed import make_embedder
    from filings_qa.ingest import ingest_ticker
    from filings_qa.store import PgVectorStore

    s = get_settings()
    client = EdgarClient(s.sec_user_agent, s.data_dir / "edgar")
    embedder = make_embedder(s.embeddings, s.embedding_model)
    store = PgVectorStore(s.database_url, dim=embedder.dim)
    for ticker in args.tickers:
        print(json.dumps(ingest_ticker(client, store, embedder, ticker.upper(), args.years)))


def _cmd_ask(args: argparse.Namespace) -> None:  # pragma: no cover - network
    from filings_qa.runtime import build_agent

    answer, evidence, _ = build_agent().ask(args.question)
    print(answer.text)
    for claim in answer.claims:
        for sid in claim.citations:
            print(f"  [{sid}] {evidence.get(sid, '(missing)')[:160]!r}")
    if answer.issues:
        print("\nVerification issues:")
        for issue in answer.issues:
            print(f"  - {issue}")


def _use_repo_root() -> None:
    """The eval/ package lives at the repo root; make it importable from the CLI."""
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())


def _cmd_build_gold(args: argparse.Namespace) -> None:  # pragma: no cover - network
    _use_repo_root()
    from eval.build_numeric_gold import main as build_gold

    build_gold(args.tickers, args.years, args.out)


def _cmd_eval(args: argparse.Namespace) -> None:  # pragma: no cover - network
    _use_repo_root()
    from eval.run_eval import main as run_eval

    run_eval(args.questions, args.out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="filings-qa")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download, chunk, embed and store 10-K filings")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--years", type=int, default=2)
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("build-gold", help="write numeric eval questions from XBRL facts")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--years", type=int, default=2)
    p.add_argument("--out", default="eval/questions_numeric.yaml")
    p.set_defaults(func=_cmd_build_gold)

    p = sub.add_parser("eval", help="run the evaluation suite")
    p.add_argument(
        "--questions",
        nargs="+",
        default=["eval/questions_numeric.yaml", "eval/questions_text.yaml"],
    )
    p.add_argument("--out-dir", default="eval/results")
    p.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
