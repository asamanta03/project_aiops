#!/usr/bin/env python3

import re
import sys
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

init(autoreset=True)

LOG_FILE = "system_logs.txt"


# ---------------------------------------------------
# Parse Log File
# ---------------------------------------------------
def parse_logs(log_file):
    log_pattern = re.compile(
        r'^(?P<timestamp>\S+ \S+) (?P<level>\w+) (?P<message>.*)$'
    )

    data = []

    try:
        with open(log_file, "r") as f:
            for line in f:
                match = log_pattern.match(line.strip())
                if match:
                    data.append(match.groupdict())
    except FileNotFoundError:
        print(Fore.RED + f"Log file not found: {log_file}")
        sys.exit(1)

    df = pd.DataFrame(data)

    if df.empty:
        print(Fore.RED + "No logs found.")
        sys.exit(1)

    return df


# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
def prepare_features(df):

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])

    # Message length
    df["message_length"] = df["message"].astype(str).apply(len)

    # Log level scoring
    level_map = {
        "INFO": 1,
        "WARNING": 2,
        "ERROR": 3,
        "CRITICAL": 4
    }

    df["level_score"] = df["level"].map(level_map)
    df["severity_score"] = df["level"].map(level_map)

    features = ["level_score", "message_length", "severity_score"]

    df = df.dropna(subset=features)

    return df, features


# ---------------------------------------------------
# Run Isolation Forest
# ---------------------------------------------------
def detect_anomalies(df, features):

    if df.shape[0] == 0:
        print(Fore.RED + "No valid data available for anomaly detection.")
        sys.exit(1)

    print(Fore.CYAN + "Training IsolationForest model...")

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    df["anomaly"] = model.fit_predict(df[features])

    return df


# ---------------------------------------------------
# Save Results
# ---------------------------------------------------
def save_results(df):

    anomalies = df[df["anomaly"] == -1]
    error_logs = df[df["level"].isin(["ERROR", "CRITICAL"])]

    df.to_csv("aiops_analyzed_logs.csv", index=False)
    anomalies.to_csv("aiops_anomaly_logs.csv", index=False)
    error_logs.to_csv("aiops_error_logs.csv", index=False)

    print(Fore.GREEN + "\nFiles Generated:")
    print("aiops_analyzed_logs.csv")
    print("aiops_anomaly_logs.csv")
    print("aiops_error_logs.csv")

    return anomalies


# ---------------------------------------------------
# Visualization
# ---------------------------------------------------
def visualize_logs(df):

    level_counts = df["level"].value_counts()

    plt.figure(figsize=(8, 5))
    level_counts.plot(kind="bar")

    plt.title("Log Level Distribution")
    plt.xlabel("Log Level")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("log_distribution.png")

    print(Fore.YELLOW + "Log distribution chart saved: log_distribution.png")


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():

    print(Fore.CYAN + "\nStarting AIOps Log Analysis...\n")

    df = parse_logs(LOG_FILE)

    print(Fore.BLUE + f"Logs loaded: {df.shape}")

    df, features = prepare_features(df)

    print(Fore.BLUE + f"Dataset after cleaning: {df.shape}")

    df = detect_anomalies(df, features)

    anomalies = save_results(df)

    print(Fore.MAGENTA + "\nSample Anomalies:")
    print(anomalies[["timestamp", "level", "message"]].head())

    visualize_logs(df)

    print(Fore.GREEN + "\nAnalysis completed.\n")


if __name__ == "__main__":
    main()
