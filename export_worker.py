from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import sys

# Allow direct script execution without package install.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exporter import export_best_model_to_onnx  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated ONNX export worker")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-basename", default=None)
    args = parser.parse_args()

    try:
        result = export_best_model_to_onnx(
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            model_basename=args.model_basename,
        )
        print(json.dumps(result))
        return 0 if result.get("success") else 1
    except Exception as e:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Unhandled ONNX worker error: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
