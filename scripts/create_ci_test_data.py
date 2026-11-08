
from pathlib import Path
import csv
import random

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

TELCO_FILE = DATA_DIR / "churn.csv"
BANK_FILE = DATA_DIR / "churn2.csv"

# Never overwrite existing local datasets.
if TELCO_FILE.exists() or BANK_FILE.exists():
    raise FileExistsError(
        "Dataset already exists. Refusing to overwrite local files."
    )

DATA_DIR.mkdir(parents=True, exist_ok=True)

columns = [
    "customerID", "gender", "SeniorCitizen", "Partner",
    "Dependents", "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

yes_no = ["Yes", "No"]

with TELCO_FILE.open("x", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for i in range(1000):
        tenure = random.randint(0, 72)
        monthly = round(random.uniform(18.0, 119.0), 2)
        total = round(tenure * monthly * random.uniform(0.85, 1.0), 2)

        if tenure == 0:
            total = 0.0

        writer.writerow({
            "customerID": f"SYNTH-{i:05d}",
            "gender": random.choice(["Male", "Female"]),
            "SeniorCitizen": random.randint(0, 1),
            "Partner": random.choice(yes_no),
            "Dependents": random.choice(yes_no),
            "tenure": tenure,
            "PhoneService": random.choice(yes_no),
            "MultipleLines": random.choice(
                ["Yes", "No", "No phone service"]
            ),
            "InternetService": random.choice(
                ["DSL", "Fiber optic", "No"]
            ),
            "OnlineSecurity": random.choice(
                ["Yes", "No", "No internet service"]
            ),
            "OnlineBackup": random.choice(
                ["Yes", "No", "No internet service"]
            ),
            "DeviceProtection": random.choice(
                ["Yes", "No", "No internet service"]
            ),
            "TechSupport": random.choice(
                ["Yes", "No", "No internet service"]
            ),
            "StreamingTV": random.choice(
                ["Yes", "No", "No internet service"]
            ),
            "StreamingMovies": random.choice(
                ["Yes", "No", "No internet service"]
            ),
            "Contract": random.choice(
                ["Month-to-month", "One year", "Two year"]
            ),
            "PaperlessBilling": random.choice(yes_no),
            "PaymentMethod": random.choice([
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]),
            "MonthlyCharges": monthly,
            "TotalCharges": f"{total:.2f}",
            "Churn": random.choice(yes_no)
        })

bank_columns = [
    "RowNumber", "CustomerId", "Surname", "CreditScore",
    "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited"
]

with BANK_FILE.open("x", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=bank_columns)
    writer.writeheader()

    for i in range(20):
        writer.writerow({
            "RowNumber": i + 1,
            "CustomerId": 100000 + i,
            "Surname": f"Synthetic{i}",
            "CreditScore": random.randint(350, 850),
            "Geography": random.choice(["France", "Spain", "Germany"]),
            "Gender": random.choice(["Male", "Female"]),
            "Age": random.randint(18, 80),
            "Tenure": random.randint(0, 10),
            "Balance": round(random.uniform(0, 200000), 2),
            "NumOfProducts": random.randint(1, 4),
            "HasCrCard": random.randint(0, 1),
            "IsActiveMember": random.randint(0, 1),
            "EstimatedSalary": round(random.uniform(10000, 200000), 2),
            "Exited": random.randint(0, 1)
        })

print("Synthetic CI datasets created successfully.")
print(f"Telco records: 1000 -> {TELCO_FILE}")
print(f"Bank records: 20 -> {BANK_FILE}")
