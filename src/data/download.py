import shutil
from pathlib import Path
import kagglehub
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DATASETS = {
    "ciciot2023": "mdabdulalemo/cic-iot-dataset2023-updated-2024-10-08",
    "edge-iiotset": "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
}

def download_all():
    DATA_DIR.mkdir(exist_ok=True)

    for name, kaggle_id in DATASETS.items():
        dest = DATA_DIR / name
        if dest.exists():
            print(f"[skip] {name} already present in {dest}")
            continue

        print(f"[download] {name} ...")
        tmp = kagglehub.dataset_download(kaggle_id)
        shutil.copytree(tmp, dest)
        print(f"[done] {name} -> {dest}")


if __name__ == "__main__":
    download_all()
