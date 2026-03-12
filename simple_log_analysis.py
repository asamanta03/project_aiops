import pandas as pd
import re
from tabulate import tabulate

# ------------------------------
# 1️⃣ Read and Parse Logs
# ------------------------------
log_file = "system_logs.txt"

log_entries = []
with open(log_file, "r") as file:
    for line in file:
        match = re.match(
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)",
            line.strip()
        )
        if match:
            timestamp, level, message = match.groups()
            log_entries.append([timestamp, level, message])

# ------------------------------
# 2️⃣ Convert to DataFrame
# ------------------------------
df = pd.DataFrame(log_entries, columns=["timestamp", "level", "message"])

# Safe datetime conversion
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

# Set timestamp as index (required for resample)
df.set_index("timestamp", inplace=True)

# ------------------------------
# 3️⃣ Detect Error Spikes (30-second window)
# ------------------------------
threshold = 3

error_df = df[df["level"] == "ERROR"]

# ✅ LOWERCASE frequency (IMPORTANT)
error_counts = error_df.resample("30s").size()

# Find time windows where error count crosses threshold
anomaly_windows = error_counts[error_counts > threshold].index

# Extract anomaly logs
anomalies = error_df[
    error_df.index.floor("30s").isin(anomaly_windows)
]

# ------------------------------
# 4️⃣ Display Anomalies
# ------------------------------
if not anomalies.empty:
    print("\n🔍 Detected Anomalies:\n")
    print(tabulate(
        anomalies.reset_index(),
        headers="keys",
        tablefmt="psql",
        showindex=False
    ))
else:
    print("\n✅ No anomalies detected (system stable).")

# ------------------------------
# 5️⃣ Display Full Logs
# ------------------------------
print("\n📜 Full Log Analysis:\n")
print(tabulate(
    df.reset_index(),
    headers="keys",
    tablefmt="psql",
    showindex=False
))
