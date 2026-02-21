from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def run_parity(models_dir: Path, contract_path: Path, onnx_path: Path) -> dict:
    # Import ONNX Runtime first in a clean process to avoid downstream DLL conflicts.
    try:
        import onnxruntime as ort
    except Exception as exc:
        return {"success": False, "error": f"onnxruntime import failed in worker: {exc}"}

    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    features = payload.get("feature_columns", [])
    input_name = payload.get("input_name", "float_input")
    if not features:
        return {"success": False, "error": "No feature_columns in contract"}

    scaler = joblib.load(models_dir / "scaler.joblib")
    metadata = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf-8"))
    ranked = sorted(
        [
            (name, float(metrics.get("R2", -1e9)))
            for name, metrics in (metadata.get("performance", {}) or {}).items()
            if name != "Ensemble"
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    if not ranked:
        return {"success": False, "error": "No ranked models in model_metadata.performance"}

    best_name = ranked[0][0]
    best_model = joblib.load(models_dir / f"model_{best_name.lower()}.joblib")

    z = np.zeros((1, len(features)), dtype=np.float64)
    if len(features) > 0:
        z[0, 0] = 0.5
    if len(features) > 1:
        z[0, 1] = -0.25
    x_raw = scaler.inverse_transform(z)

    model_input_df = pd.DataFrame(x_raw, columns=features)
    needs_scaled_input = best_name in ["NeuralNetwork", "Ridge", "ElasticNet", "SVR"]
    if needs_scaled_input:
        x_scaled = scaler.transform(model_input_df)
        py_pred = float(best_model.predict(x_scaled)[0])
    else:
        py_pred = float(best_model.predict(model_input_df)[0])

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input = x_scaled.astype(np.float32) if needs_scaled_input else x_raw.astype(np.float32)
    onnx_pred = float(sess.run(None, {input_name: onnx_input})[0].ravel()[0])
    delta = abs(py_pred - onnx_pred)
    return {
        "success": True,
        "best_model": best_name,
        "onnx_input_mode": ("scaled" if needs_scaled_input else "raw"),
        "python_prediction": py_pred,
        "onnx_prediction": onnx_pred,
        "abs_delta": delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated parity check worker")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--contract-file", required=True)
    parser.add_argument("--onnx-file", required=True)
    args = parser.parse_args()

    try:
        result = run_parity(
            models_dir=Path(args.models_dir),
            contract_path=Path(args.contract_file),
            onnx_path=Path(args.onnx_file),
        )
        print(json.dumps(result))
        return 0 if result.get("success") else 1
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"Unhandled parity worker error: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
