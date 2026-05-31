# Census Income Minimal API

This folder keeps only the pieces needed to train and serve three pickle models:

- `models/lr_model.pkl`
- `models/gbt_model.pkl`
- `models/xgb_model.pkl`

Train:

```powershell
.\.venv\Scripts\python.exe census_income_minimal\train_lr.py
.\.venv\Scripts\python.exe census_income_minimal\train_gbt.py
.\.venv\Scripts\python.exe census_income_minimal\train_xgb.py
```

Run API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Send predictions to `POST /predict` with `model` as `lr`, `gbt`, or `xgb`.
