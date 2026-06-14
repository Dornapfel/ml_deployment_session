import argparse
import requests
import sys

BASE_URL = "http://127.0.0.1:5000"

VALID_DATA = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 50000,
    "BILL_AMT2": 50000,
    "BILL_AMT3": 50000,
    "BILL_AMT4": 50000,
    "BILL_AMT5": 50000,
    "BILL_AMT6": 50000,
    "PAY_AMT1": 0,
    "PAY_AMT2": 0,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
}

def health():
    r = requests.get(f"{BASE_URL}/health")
    print(r.status_code)
    print(r.json())

def predict():
    r = requests.post(f"{BASE_URL}/predict", json=VALID_DATA)
    print(r.status_code)
    print(r.json())

def run_tests():
    print("Running API tests...\n")

    # health test
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("health: OK")

    # predict test
    r = requests.post(f"{BASE_URL}/predict", json=VALID_DATA)
    res = r.json()
    assert r.status_code == 200
    assert "prediction" in res
    print("predict: OK")

    # missing fields test
    r = requests.post(f"{BASE_URL}/predict", json={"LIMIT_BAL": 50000})
    assert r.status_code == 400
    print("missing fields: OK")

    # invalid data test
    r = requests.post(f"{BASE_URL}/predict", data="bad")
    assert r.status_code in (400, 415)
    print("invalid data: OK")

    print("\nAll tests passed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["health", "predict", "test"])
    args = parser.parse_args()

    if args.command == "health":
        health()
    elif args.command == "predict":
        predict()
    elif args.command == "test":
        run_tests()

if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        print("TEST FAILED")
        sys.exit(1)
