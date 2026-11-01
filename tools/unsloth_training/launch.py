#!/usr/bin/env python3
"""Start Recommendation only after verified Response completion and Q gates."""
import argparse
from pathlib import Path
import subprocess
import sys

import production


def verified_response(path):
    return production.verify_response_completion(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-output", type=Path, required=True)
    parser.add_argument("--recommendation-output", type=Path, required=True)
    parser.add_argument("--recommendation-benchmark-result", type=Path, required=True)
    parser.add_argument("--recommendation-smoke-output", type=Path, required=True)
    parser.add_argument("--recommendation-loss-evidence", type=Path, required=True)
    parser.add_argument("--recommendation-batch", type=int, choices=(1,), required=True)
    args = parser.parse_args(argv)
    verified_response(args.response_output.absolute())
    production.verify_benchmark_result(args.recommendation_benchmark_result.absolute(), "recommendation")
    production.verify_smoke_output(args.recommendation_smoke_output.absolute(), "recommendation")
    command = [sys.executable, str(Path(production.__file__).resolve()),
               "--mode", "train", "--agent", "recommendation",
               "--output", str(args.recommendation_output.absolute()),
               "--batch", str(args.recommendation_batch),
               "--benchmark-result", str(args.recommendation_benchmark_result.absolute()),
               "--smoke-output", str(args.recommendation_smoke_output.absolute()),
               "--loss-evidence", str(args.recommendation_loss_evidence.absolute()),
               "--response-output", str(args.response_output.absolute())]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
