# Census Databricks Workspace

This folder is organized by file purpose:

- `data/raw/` - source CSV files
- `docs/` - problem statements and reference documents
- `notebooks/` - Databricks and local notebook work
- `reports/` - generated HTML reports
- `scripts/` - local utility scripts
- `scripts/models/` - Databricks model scripts
- `prototypes/` - standalone HTML prototypes
- `my_venv/` - local Python virtual environment

To regenerate the profiling report, run:

```powershell
.\my_venv\Scripts\python.exe .\scripts\generate_dataset_report.py
```
