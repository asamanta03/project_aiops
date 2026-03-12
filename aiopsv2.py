#!/usr/bin/env python3

"""
Enterprise AIOps Engine
Author: DevOps Architect Implementation

Features
--------
• Log anomaly detection
• Metric anomaly detection
• Root cause analysis
• Incident correlation
• Auto remediation
• Self-healing verification
• Failure prediction
"""

import pandas as pd
import numpy as np
import subprocess
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

LOG_FILE = "system_logs.txt"
CPU_THRESHOLD = 85
MEM_THRESHOLD = 90

logging.basicConfig(
    filename="aiops_engine.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------------------------------
# Log Parser
# -------------------------------------------------------

def parse_logs(file):

    data = []

    with open(file) as f:
        for line in f:
            parts = line.strip().split(" ", 3)
            if len(parts) < 4:
                continue

            timestamp = parts[0] + " " + parts[1]
            level = parts[2]
            msg = parts[3]

            data.append([timestamp, level, msg])

    df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


# -------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------

def feature_engineering(df):

    level_map = {
        "INFO": 1,
        "WARNING": 2,
        "ERROR": 3,
        "CRITICAL": 4
    }

    df["level_score"] = df["level"].map(level_map).fillna(0)
    df["msg_len"] = df["message"].apply(len)

    def severity(msg):
        msg = msg.lower()

        if "panic" in msg or "kernel" in msg:
            return 5
        elif "failed" in msg or "error" in msg:
            return 4
        elif "timeout" in msg or "slow" in msg:
            return 3
        else:
            return 1

    df["severity"] = df["message"].apply(severity)

    return df


# -------------------------------------------------------
# Anomaly Detection (Logs)
# -------------------------------------------------------

def detect_log_anomaly(df):

    model = IsolationForest(contamination=0.05)

    df["anomaly"] = model.fit_predict(
        df[["level_score", "msg_len", "severity"]]
    )

    df["status"] = df["anomaly"].apply(
        lambda x: "ANOMALY" if x == -1 else "NORMAL"
    )

    return df


# -------------------------------------------------------
# Metric Collection
# -------------------------------------------------------

def collect_metrics():

    cpu = float(subprocess.getoutput(
        "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'"
    ))

    mem = float(subprocess.getoutput(
        "free | awk '/Mem/ {print $3/$2 * 100.0}'"
    ))

    disk = float(subprocess.getoutput(
        "df / | tail -1 | awk '{print $5}' | sed 's/%//'"
    ))

    return cpu, mem, disk


# -------------------------------------------------------
# Predictive Failure Detection
# -------------------------------------------------------

def predict_failure(history):

    if len(history) < 5:
        return False

    X = np.arange(len(history)).reshape(-1, 1)
    y = np.array(history)

    model = LinearRegression()
    model.fit(X, y)

    future = model.predict([[len(history) + 3]])

    if future > CPU_THRESHOLD:
        return True

    return False


# -------------------------------------------------------
# Root Cause Analysis
# -------------------------------------------------------

def root_cause(df):

    errors = df[df["level"] == "ERROR"]

    if len(errors) == 0:
        return "Unknown"

    common = errors["message"].value_counts().idxmax()

    return common


# -------------------------------------------------------
# Auto Remediation Engine
# -------------------------------------------------------

def remediation(action):

    logging.warning(f"Executing remediation: {action}")

    try:

        if action == "restart_service":

            subprocess.run(["systemctl", "restart", "nginx"])

        elif action == "clear_tmp":

            subprocess.run(["rm", "-rf", "/tmp/*"])

        elif action == "scale_kubernetes":

            subprocess.run([
                "kubectl",
                "scale",
                "deployment",
                "web",
                "--replicas=5"
            ])

    except Exception as e:

        logging.error(e)


# -------------------------------------------------------
# Self Healing Verification
# -------------------------------------------------------

def verify():

    cpu, mem, disk = collect_metrics()

    if cpu < CPU_THRESHOLD and mem < MEM_THRESHOLD:
        return True

    return False


# -------------------------------------------------------
# Incident Correlation
# -------------------------------------------------------

def correlate_incidents(df):

    grouped = df.groupby("severity").size()

    return grouped


# -------------------------------------------------------
# Main AIOps Pipeline
# -------------------------------------------------------

def main():

    print("\n===== AIOps Engine Started =====\n")

    df = parse_logs(LOG_FILE)

    df = feature_engineering(df)

    df = detect_log_anomaly(df)

    anomalies = df[df["status"] == "ANOMALY"]

    print(tabulate(anomalies.head(10), headers="keys"))

    root = root_cause(df)

    print(f"\nRoot Cause Candidate: {root}")

    cpu, mem, disk = collect_metrics()

    print(f"\nCPU: {cpu}  MEM: {mem}  DISK: {disk}")

    history = [cpu]

    if predict_failure(history):

        print("\n⚠️ Predictive Alert: CPU spike expected")

    if cpu > CPU_THRESHOLD:

        remediation("scale_kubernetes")

    if mem > MEM_THRESHOLD:

        remediation("restart_service")

    if verify():

        print("\n✅ Self Healing Successful")

    else:

        print("\n❌ System still unhealthy")

    correlation = correlate_incidents(df)

    print("\nIncident Correlation")
    print(correlation)

    df.to_csv("aiops_report.csv")


if __name__ == "__main__":
    main()
