#!/usr/bin/env python3
"""Run the cue eval on SageMaker (stub).

TODO:
  - Serialize prompts for every item × channel × cue type to JSONL and upload to S3.
  - Submit an async job (SageMakerBackend) with a custom inference.py that returns
    completions + last-prompt-token hidden states.
  - Poll for completion, download outputs to results/<model>/, then reuse
    cue_channels.eval.runner.summarize on the downloaded records.
"""

from __future__ import annotations

import argparse

from cue_channels.config import DEFAULT_MODEL, MODEL_REGISTRY, RunConfig
from cue_channels.models import make_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--instance-type", default="ml.g5.xlarge")
    args = ap.parse_args()

    backend = make_backend(RunConfig(model=args.model, backend="sagemaker"))
    backend.load()  # raises NotImplementedError until the stub is filled in


if __name__ == "__main__":
    main()
