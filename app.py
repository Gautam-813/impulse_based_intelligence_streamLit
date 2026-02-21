from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_feature_engineering import AdvancedFeatureEngineer  # noqa: E402
from ml_models import AdvancedMLModels  # noqa: E402
from onnx_export_app import export_best_model_to_onnx_isolated  # noqa: E402

# Feature set aligned with current MT5 EA feature mapper (MLIntelligence.mqh).
MT5_EA_FEATURE_COLUMNS = [
    "Impulse%",
    "Price_Movement_Pct",
    "Hour",
    "DayOfWeek",
    "Month",
    "Quarter",
    "Minutes_Since_Midnight",
    "Tokyo_London_Overlap",
    "London_NY_Overlap",
    "NY_Sydney_Overlap",
    "Hour_Sin",
    "Hour_Cos",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "SMA_5",
    "SMA_20",
    "EMA_12",
    "EMA_26",
    "Price_Above_SMA5",
    "Price_Above_SMA20",
    "Price_SMA5_Distance",
    "Price_SMA20_Distance",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "RSI",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "BB_Position",
    "Price_Volatility",
    "Trend_Strength",
    "Market_Efficiency",
    "Impulse_Momentum",
    "Recent_Avg_Reversal",
    "Recent_Max_Reversal",
    "Recent_Min_Reversal",
    "Recent_Reversal_Std",
    "Time_Since_Last_Similar",
    "Impulse_RSI_Interaction",
    "Direction_Trend_Interaction",
    "Direction_Numeric",
]


def normalize_training_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common dataset schema variants before feature engineering."""
    out = df.copy()
    if "BasePrice" not in out.columns and "EntryPrice" in out.columns:
        out["BasePrice"] = out["EntryPrice"]
    if "Reversal%" not in out.columns and {"Pullback", "Impulse"}.issubset(out.columns):
        out["Reversal%"] = np.where(
            out["Impulse"].replace(0, np.nan).notna(),
            (out["Pullback"] / out["Impulse"].replace(0, np.nan)) * 100.0,
            0.0,
        )
        out["Reversal%"] = out["Reversal%"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def validate_export_artifacts(contract_path: Path, onnx_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "onnx_exists": onnx_path.exists(),
        "contract_exists": contract_path.exists(),
        "feature_count": None,
        "scaler_present": False,
        "scaler_mean_len": 0,
        "scaler_scale_len": 0,
        "selected_model": None,
        "status_ok": False,
        "error": "",
    }
    if not result["onnx_exists"] or not result["contract_exists"]:
        result["error"] = "Missing ONNX or feature contract file."
        return result

    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        features = payload.get("feature_columns", [])
        scaler = payload.get("scaler", {}) or {}
        result["feature_count"] = len(features)
        result["selected_model"] = payload.get("selected_model")
        result["scaler_present"] = bool(scaler.get("present"))
        result["scaler_mean_len"] = len(scaler.get("mean", []))
        result["scaler_scale_len"] = len(scaler.get("scale", []))
        if result["scaler_present"] and (
            result["scaler_mean_len"] != result["feature_count"]
            or result["scaler_scale_len"] != result["feature_count"]
        ):
            result["error"] = "Scaler mean/scale length mismatch with feature_count."
            return result
        result["status_ok"] = True
        return result
    except Exception as exc:
        result["error"] = f"Validation failed: {exc}"
        return result


def parity_sanity_check(models_dir: Path, contract_path: Path, onnx_path: Path) -> Dict[str, Any]:
    """
    Compare best-model prediction and ONNX prediction on one synthetic sample.
    Uses ONNX Runtime when available.
    """
    worker = Path(__file__).with_name("parity_worker.py")
    cmd = [
        sys.executable,
        str(worker),
        "--models-dir",
        str(models_dir),
        "--contract-file",
        str(contract_path),
        "--onnx-file",
        str(onnx_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        return {"success": False, "error": f"Failed to start parity worker: {exc}"}

    raw = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    try:
        result = json.loads(raw.splitlines()[-1] if "\n" in raw else raw)
    except Exception:
        return {"success": False, "error": f"Parity worker returned invalid output: {raw[:400]}"}
    return result


def train_models(
    data_csv: Path,
    models_dir: Path,
    mt5_compatible_only: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    try:
        df = pd.read_csv(data_csv)
    except Exception as exc:
        return False, f"Failed to read CSV: {exc}", {}

    df = normalize_training_df(df)
    required_cols = {"Time", "Direction", "Impulse", "Pullback", "BasePrice"}
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        return False, f"Missing required columns after normalization: {missing}", {}

    engineer = AdvancedFeatureEngineer()
    df_ml, feature_cols = engineer.prepare_ml_dataset(df)

    if mt5_compatible_only:
        missing = [f for f in MT5_EA_FEATURE_COLUMNS if f not in df_ml.columns]
        if missing:
            return False, f"Missing MT5-compatible features in engineered dataset: {missing}", {}
        feature_cols = [f for f in MT5_EA_FEATURE_COLUMNS if f in feature_cols]
        if len(feature_cols) != len(MT5_EA_FEATURE_COLUMNS):
            return False, "Failed to lock MT5-compatible feature list exactly.", {}

    ml_system = AdvancedMLModels()
    results = ml_system.train_complete_system(df_ml, feature_cols)

    models_dir.mkdir(parents=True, exist_ok=True)
    ml_system.save_models(str(models_dir) + "/")
    return True, "Training completed", {
        "feature_count": len(feature_cols),
        "sample_count": len(df_ml),
        "mt5_compatible_only": mt5_compatible_only,
        "performance": results.get("performance", {}),
    }


def main() -> None:
    st.set_page_config(page_title="ONNX Exporter for MT5", layout="wide")
    st.title("Standalone ONNX Exporter (MT5)")
    st.caption("Trains model from impulse_based_intelligence data and exports MT5-ready ONNX + feature contract.")

    default_data = ROOT_DIR / "data" / "Impulse_Reversal.csv"
    default_models_dir = ROOT_DIR / "models"
    default_onnx_dir = ROOT_DIR / "onnx_exports"

    with st.sidebar:
        st.subheader("Paths")
        data_path = Path(st.text_input("Training CSV (path)", str(default_data)))
        uploaded_csv = st.file_uploader("Or upload training CSV", type=["csv"])

        st.markdown("#### Save Locations")
        models_dir = Path(st.text_input("Models output dir", str(default_models_dir)))
        onnx_output_dir = Path(st.text_input("ONNX output dir", str(default_onnx_dir)))
        model_basename = st.text_input("ONNX base filename (optional)", "")
        mt5_compatible_only = st.checkbox(
            "MT5-compatible feature set only (recommended)",
            value=True,
            help="Use only features currently supported by MA_Distance_Grid_EA.mq5 + MLIntelligence.mqh mapper.",
        )
        run_parity_check = st.checkbox("Run Python vs ONNX parity sanity check", value=True)

    col_a, col_b = st.columns(2)
    with col_a:
        train_clicked = st.button("1) Train / Refresh Models", use_container_width=True)
    with col_b:
        export_clicked = st.button("2) Export ONNX (Isolated Worker)", use_container_width=True)

    if train_clicked:
        if uploaded_csv is not None:
            upload_dir = ROOT_DIR / "onnx_export_app" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            data_path = upload_dir / uploaded_csv.name
            data_path.write_bytes(uploaded_csv.getvalue())
            st.info(f"Using uploaded CSV: {data_path}")

        models_dir.mkdir(parents=True, exist_ok=True)
        onnx_output_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Training models..."):
            ok, msg, info = train_models(
                data_path,
                models_dir,
                mt5_compatible_only=mt5_compatible_only,
            )
        if not ok:
            st.error(msg)
        else:
            st.success(msg)
            st.json(info)

    if export_clicked:
        models_dir.mkdir(parents=True, exist_ok=True)
        onnx_output_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Exporting ONNX in isolated worker..."):
            result = export_best_model_to_onnx_isolated(
                models_dir=str(models_dir),
                output_dir=str(onnx_output_dir),
                model_basename=model_basename.strip() or None,
            )

        if not result.get("success"):
            st.error(result.get("error", "Unknown ONNX export error"))
            return

        onnx_path = Path(result["onnx_file"])
        contract_path = Path(result["feature_contract_file"])
        validation = validate_export_artifacts(contract_path, onnx_path)

        st.success(
            f"ONNX export completed. Model={result.get('selected_model')} "
            f"| Features={result.get('feature_count')}"
        )
        st.markdown("### Export Status")
        st.code(
            "\n".join(
                [
                    f"ONNX file: {onnx_path}",
                    f"Exists: {'YES' if validation['onnx_exists'] else 'NO'}",
                    "",
                    f"Feature contract: {contract_path}",
                    f"Exists: {'YES' if validation['contract_exists'] else 'NO'}",
                    "",
                    f"Selected model: {validation.get('selected_model')}",
                    f"Feature count: {validation.get('feature_count')}",
                    f"Scaler present: {validation.get('scaler_present')}",
                    f"Scaler mean len: {validation.get('scaler_mean_len')}",
                    f"Scaler scale len: {validation.get('scaler_scale_len')}",
                    f"Validation status: {'OK' if validation.get('status_ok') else 'FAILED'}",
                    f"Error: {validation.get('error') or '-'}",
                    f"Generated at: {datetime.now().isoformat()}",
                ]
            )
        )

        if onnx_path.exists():
            st.download_button(
                "Download ONNX file",
                data=onnx_path.read_bytes(),
                file_name=onnx_path.name,
                mime="application/octet-stream",
                use_container_width=True,
            )
        if contract_path.exists():
            st.download_button(
                "Download feature contract JSON",
                data=contract_path.read_text(encoding="utf-8"),
                file_name=contract_path.name,
                mime="application/json",
                use_container_width=True,
            )

        if run_parity_check and validation.get("status_ok"):
            st.markdown("### Parity Sanity Check")
            parity = parity_sanity_check(models_dir, contract_path, onnx_path)
            if parity.get("success"):
                st.json(parity)
            else:
                st.warning(parity.get("error", "Parity sanity check failed"))
                if "python_prediction" in parity:
                    st.write(f"Python prediction: {parity['python_prediction']}")

    st.markdown("---")
    st.markdown(
        "MT5 usage: Copy generated `.onnx` and `.feature_contract.json` to `MQL5/Files`, "
        "set EA `Use_ONNX_In_Tester=true` and `ONNX_Model_File=<filename>.onnx`."
    )


if __name__ == "__main__":
    main()
