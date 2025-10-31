import pandas as pd
from evidently.presets import DataDriftPreset
from evidently import Report

def monitor_drift():
    reference = pd.read_csv('data/processed/train.csv')
    current = pd.read_csv('data/processed/test.csv')

    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=reference, current_data=current)
    my_eval.save_html('output/data_drift/data_drift_report.html')
    print("Data drift report saved as HTML.")

if __name__ == "__main__":
    monitor_drift()
