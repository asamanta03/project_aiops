import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import Fore, Style

# ------------------------------
# 1️⃣ Read and Parse Logs
# ------------------------------
log_file_path = "system_logs.txt"  # Update with your log file path

try:
    with open(log_file_path, "r") as file:
        logs = file.readlines()
except FileNotFoundError:
    print(Fore.RED + f"❌ Log file not found: {log_file_path}" + Style.RESET_ALL)
    exit()

data = []
for log in logs:
    parts = log.strip().split(" ", 3)
    if len(parts) < 4:
        continue
    timestamp = parts[0] + " " + parts[1]
    level = parts[2]
    message = parts[3]
    data.append([timestamp, level, message])

df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df.dropna(subset=["timestamp"], inplace=True)

# ------------------------------
# 2️⃣ Feature Engineering
# ------------------------------
level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
df["level_score"] = df["level"].map(level_mapping).fillna(0)
df["message_length"] = df["message"].apply(len)

# Simple severity scoring based on message content
def detect_severity(msg):
    msg_lower = msg.lower()
    if any(word in msg_lower for word in ["failed", "error", "critical", "panic"]):
        return 4
    elif any(word in msg_lower for word in ["warning", "slow", "timeout"]):
        return 3
    else:
        return 1

df["severity_score"] = df["message"].apply(detect_severity)

# ------------------------------
# 3️⃣ AI Model - Isolation Forest
# ------------------------------
model = IsolationForest(contamination=0.08, random_state=42)
df["anomaly"] = model.fit_predict(df[["level_score", "message_length", "severity_score"]])
df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

# ------------------------------
# 4️⃣ Display Results
# ------------------------------
anomalies = df[df["is_anomaly"] == "❌ Anomaly"]

print(Fore.CYAN + "\n🔍 Detected Log Anomalies:\n" + Style.RESET_ALL)
print(tabulate(
    anomalies[["timestamp", "level", "message", "severity_score", "message_length", "is_anomaly"]],
    headers="keys",
    tablefmt="psql",
    showindex=False
))

# ------------------------------
# 5️⃣ Save & Visualize
# ------------------------------
# Save anomalies to CSV
csv_file = f"anomalies_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
anomalies.to_csv(csv_file, index=False)
print(Fore.YELLOW + f"\n📁 Anomalies report saved as: {csv_file}" + Style.RESET_ALL)

# Generate error frequency chart
plt.figure(figsize=(8, 4))
df["level"].value_counts().plot(kind="bar", color=["green", "orange", "red", "purple"])
plt.title("Log Level Frequency")
plt.xlabel("Log Level")
plt.ylabel("Count")
plt.tight_layout()
chart_file = "log_level_frequency.png"
plt.savefig(chart_file)
print(Fore.GREEN + f"📊 Chart saved as: {chart_file}" + Style.RESET_ALL)

# ------------------------------
# 6️⃣ Summary
# ------------------------------
print(Fore.MAGENTA + "\n📈 Summary Report" + Style.RESET_ALL)
print(f"Total Logs: {len(df)}")
print(f"Detected Anomalies: {len(anomalies)}")
print(f"Normal Logs: {len(df) - len(anomalies)}")
print(Fore.CYAN + "\n✅ Analysis complete!" + Style.RESET_ALL)
