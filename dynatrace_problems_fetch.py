#!/usr/bin/env python3

import requests
import pandas as pd
import os

# Dynatrace environment
DYNATRACE_URL = "https://YOUR_ENV.live.dynatrace.com"
API_TOKEN = os.getenv("DYNATRACE_API_TOKEN")

url = f"{DYNATRACE_URL}/api/v2/problems"

headers = {
    "Authorization": f"Api-Token {API_TOKEN}",
    "Content-Type": "application/json"
}

params = {
    "pageSize": 100
}

response = requests.get(url, headers=headers, params=params)

if response.status_code != 200:
    print("Failed to fetch problems:", response.text)
    exit()

data = response.json()

records = []

for problem in data.get("problems", []):
    records.append({
        "problem_id": problem.get("problemId"),
        "title": problem.get("title"),
        "impact_level": problem.get("impactLevel"),
        "severity": problem.get("severityLevel"),
        "status": problem.get("status"),
        "start_time": problem.get("startTime")
    })

df = pd.DataFrame(records)

os.makedirs("dynatrace", exist_ok=True)

df.to_csv("dynatrace/dynatrace_problems.csv", index=False)

print("Dynatrace problems exported to dynatrace_problems.csv")
