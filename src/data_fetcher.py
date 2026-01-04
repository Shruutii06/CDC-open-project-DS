import os
import time
import requests
import pandas as pd
from tqdm import tqdm
import json

# ================== CONFIG ==================

MAPBOX_TOKEN = "pk.eyJ1Ijoic2hydXRpNjQ1IiwiYSI6ImNtamNqZnpkYTBrNmIzY3F1Yzg4OXRxbG4ifQ.jsgYqvFoNazyZinBIVvAAg"

ZOOM = 18
IMG_SIZE = "224x224"

TRAIN_IMG_DIR = "data/images/train"
TEST_IMG_DIR  = "data/images/test"

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)

# ================== FUNCTIONS ==================

def download_satellite_image(lat, lon, save_path):
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon},{lat},{ZOOM}/{IMG_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed [{response.status_code}] for {lat}, {lon}")
            return False

    except Exception as e:
        print("Request error:", e)
        return False


def fetch_images(df, img_dir, id_col="id"):
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        img_id = row[id_col]
        lat = float(row["lat"])
        lon = float(row["long"])

        path = os.path.join(img_dir, f"{img_id}.png")

        if os.path.exists(path):
            continue

        ok = download_satellite_image(lat, lon, path)

        if not ok:
            failed.append(int(img_id))

        time.sleep(0.05)  # safe rate limit

    return failed


# ================== MAIN ==================

if __name__ == "__main__":

    train_df = pd.read_csv("data/raw/train.csv")
    test_df  = pd.read_csv("data/raw/test.csv")

    # 🔍 COLUMN SAFETY CHECK
    required_cols = {"id", "lat", "long"}
    assert required_cols.issubset(train_df.columns), f"Missing columns in train: {train_df.columns}"
    assert required_cols.issubset(test_df.columns),  f"Missing columns in test: {test_df.columns}"

    print("Downloading TRAIN images...")
    train_failed = fetch_images(train_df, TRAIN_IMG_DIR)

    print("Downloading TEST images...")
    test_failed = fetch_images(test_df, TEST_IMG_DIR)

    with open("failed_train.json", "w") as f:
        json.dump(train_failed, f)

    with open("failed_test.json", "w") as f:
        json.dump(test_failed, f)

    print(f"Train failed: {len(train_failed)}")
    print(f"Test failed : {len(test_failed)}")
