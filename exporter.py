from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib


def _rank_models(metadata: Dict[str, Any]) -> List[str]:
    performance = metadata.get("performance", {}) or {}
    ranked: List[Tuple[str, float]] = []
    for name, metrics in performance.items():
        if name == "Ensemble":
            continue
        try:
            ranked.append((name, float(metrics.get("R2", -1e9))))
        except Exception:
            ranked.append((name, -1e9))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked]


def _model_file_for_name(name: str) -> str:
    return f"model_{name.lower()}.joblib"


def export_best_model_to_onnx(
    models_dir: str,
    output_dir: str,
    model_basename: str | None = None,
) -> Dict[str, Any]:
    models_path = Path(models_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata_file = models_path / "model_metadata.json"
    if not metadata_file.exists():
        return {"success": False, "error": f"Missing metadata: {metadata_file}"}

    with metadata_file.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        return {
            "success": False,
            "error": (
                "ONNX export dependencies missing. Install: "
                "pip install onnx skl2onnx onnxconverter-common. "
                f"Details: {e}"
            ),
        }

    ranked_models = _rank_models(metadata)
    if not ranked_models:
        return {"success": False, "error": "No ranked models in model_metadata.performance"}

    selected_name = None
    selected_model = None
    for name in ranked_models:
        model_file = models_path / _model_file_for_name(name)
        if not model_file.exists():
            continue
        try:
            selected_model = joblib.load(model_file)
            selected_name = name
            break
        except Exception:
            continue

    if selected_model is None or selected_name is None:
        return {"success": False, "error": "No exportable model file found in models directory"}

    scaler_file = models_path / "scaler.joblib"
    scaler_payload: Dict[str, Any] = {"present": False}
    scaler = None
    if scaler_file.exists():
        try:
            scaler = joblib.load(scaler_file)
            scaler_payload = {
                "present": True,
                "mean": getattr(scaler, "mean_", []).tolist() if hasattr(scaler, "mean_") else [],
                "scale": getattr(scaler, "scale_", []).tolist() if hasattr(scaler, "scale_") else [],
                "var": getattr(scaler, "var_", []).tolist() if hasattr(scaler, "var_") else [],
            }
        except Exception:
            scaler_payload = {"present": False}
            scaler = None

    # Strict feature ordering priority: scaler -> model -> metadata fallback.
    feature_cols: List[str] = []
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        feature_cols = [str(x) for x in list(scaler.feature_names_in_)]
    elif hasattr(selected_model, "feature_names_in_"):
        feature_cols = [str(x) for x in list(selected_model.feature_names_in_)]
    else:
        feature_cols = list((metadata.get("feature_importance") or {}).keys())

    if not feature_cols:
        return {"success": False, "error": "No feature columns available from scaler/model/metadata"}

    n_features = len(feature_cols)

    if scaler_payload.get("present"):
        mean_len = len(scaler_payload.get("mean", []))
        scale_len = len(scaler_payload.get("scale", []))
        if mean_len != n_features or scale_len != n_features:
            return {
                "success": False,
                "error": (
                    "Scaler length mismatch: feature count does not match scaler mean/scale lengths "
                    f"({n_features} vs mean={mean_len}, scale={scale_len})"
                ),
            }

    initial_types = [("float_input", FloatTensorType([None, n_features]))]

    try:
        onnx_model = convert_sklearn(selected_model, initial_types=initial_types, target_opset=15)
    except Exception as e:
        return {"success": False, "error": f"Failed ONNX conversion for '{selected_name}': {e}"}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = model_basename or f"mt5_onnx_{selected_name.lower()}_{stamp}"

    onnx_file = out_path / f"{base}.onnx"
    contract_file = out_path / f"{base}.feature_contract.json"

    with onnx_file.open("wb") as f:
        f.write(onnx_model.SerializeToString())

    contract_payload = {
        "created_at": datetime.now().isoformat(),
        "selected_model": selected_name,
        "r2_score": float((metadata.get("performance", {}).get(selected_name, {}) or {}).get("R2", 0.0)),
        "input_name": "float_input",
        "input_shape": ["batch", n_features],
        "output_notes": "Single regression output: exit_reversal_percent",
        "feature_columns": feature_cols,
        "scaler": scaler_payload,
        "mt5_notes": {
            "data_type": "float32",
            "order_is_strict": True,
            "normalize_in_mt5_exactly_like_python": True,
        },
    }

    with contract_file.open("w", encoding="utf-8") as f:
        json.dump(contract_payload, f, indent=2)

    return {
        "success": True,
        "selected_model": selected_name,
        "onnx_file": str(onnx_file),
        "feature_contract_file": str(contract_file),
        "feature_count": n_features,
    }


def export_best_model_to_onnx_isolated(
    models_dir: str,
    output_dir: str,
    model_basename: str | None = None,
    python_executable: str | None = None,
) -> Dict[str, Any]:
    python_executable = python_executable or sys.executable
    worker_file = Path(__file__).with_name("export_worker.py")
    cmd = [
        python_executable,
        str(worker_file),
        "--models-dir",
        models_dir,
        "--output-dir",
        output_dir,
    ]
    if model_basename:
        cmd.extend(["--model-basename", model_basename])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return {"success": False, "error": f"Failed to start ONNX worker process: {e}"}

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr or stdout or f"worker exit code {proc.returncode}"
        return {"success": False, "error": f"ONNX worker failed: {details}"}

    stdout = (proc.stdout or "").strip()
    try:
        result = json.loads(stdout.splitlines()[-1] if "\n" in stdout else stdout)
    except Exception:
        return {"success": False, "error": f"ONNX worker returned invalid JSON. Raw output: {stdout[:400]}"}

    return result
