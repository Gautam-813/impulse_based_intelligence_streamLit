# Standalone ONNX Export App

This folder provides a dedicated Streamlit workflow to:

1. Train models from `impulse_based_intelligence/data/Impulse_Reversal.csv`
2. Export the best model to MT5-ready ONNX
3. Generate a strict feature contract JSON for MT5 feature ordering and scaler parity
4. Run an optional Python vs ONNX parity sanity check

## Run

From `D:\backtester\backtester-17-02-2026-evening\impulse_based_intelligence`:

```bash
streamlit run onnx_export_app/app.py
```

## Outputs

By default, the app writes:

- models: `impulse_based_intelligence/models`
- ONNX exports: `impulse_based_intelligence/onnx_exports`

Expected artifacts:

- `<model_name>.onnx`
- `<model_name>.feature_contract.json`

## MT5 Strategy Tester usage

Copy both files into MT5:

- `MQL5/Files/<model_name>.onnx`
- `MQL5/Files/<model_name>.feature_contract.json`

Then in EA inputs:

- `Use_ONNX_In_Tester = true`
- `ONNX_Model_File = "<model_name>.onnx"`

## Important for MT5 parity

Keep `MT5-compatible feature set only` enabled in the Streamlit sidebar.
This ensures the trained/exported ONNX model uses only features the current
`MA_Distance_Grid_EA.mq5` + `MLIntelligence.mqh` mapping can provide.

## Notes

- Export runs in an isolated worker process to reduce Windows DLL conflicts.
- Contract generation enforces strict feature/scaler length parity and fails fast if mismatched.
- Parity sanity check requires `onnxruntime`.
