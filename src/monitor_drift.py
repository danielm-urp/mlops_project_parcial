import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

def monitor_drift():
    reference = pd.read_csv('data/processed/train.csv')
    current = pd.read_csv('data/processed/test.csv')

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html('data_drift_report.html')
    print("Data drift report saved as HTML.")

if __name__ == "__main__":
    monitor_drift()
