import pandas as pd
from ydata_profiling import ProfileReport
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "census_data_raw.csv"
REPORT_PATH = ROOT_DIR / "reports" / "dataset_report.html"

data = pd.read_csv(DATA_PATH)

profile = ProfileReport(
    data,
    title="Dataset Analysis",
    explorative=True
)

profile.to_file(REPORT_PATH)
