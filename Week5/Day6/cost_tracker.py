import pandas as pd

usd_to_inr = 86

logs = [
    {"tokens": 850, "model": "GPT-4o"},
    {"tokens": 1200, "model": "GPT-4o"},
    {"tokens": 600, "model": "GPT-4o mini"},
    {"tokens": 5000, "model": "GPT-4o"},
]

prices = {
    "GPT-4o": 5,
    "GPT-4o mini": 0.15
}

rows = []

for log in logs:

    usd = (log["tokens"] / 1_000_000) * prices[log["model"]]
    inr = usd * usd_to_inr

    rows.append({
        "Model": log["model"],
        "Tokens": log["tokens"],
        "Cost USD": round(usd, 6),
        "Cost INR": round(inr, 4)
    })

df = pd.DataFrame(rows)

print(df)

print("\nDaily Total INR:", round(df["Cost INR"].sum(), 4))

df.to_csv("cost_audit.csv", index=False)

print("\nSaved cost_audit.csv")