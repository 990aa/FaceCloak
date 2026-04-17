import csv
import os
import requests
from pathlib import Path

FIXTURES_DIR = Path("tests/fixtures/general")
MANIFEST_PATH = Path("benchmarks/general_manifest.csv")

IMAGES = {
    "cat_1": "https://picsum.photos/id/40/400/400",
    "dog_1": "https://picsum.photos/id/237/400/400",
    "car_1": "https://picsum.photos/id/1071/400/400",
    "building_1": "https://picsum.photos/id/122/400/400",
    "landscape_1": "https://picsum.photos/id/1018/400/400",
    "document_1": "https://picsum.photos/id/366/400/400"
}

def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    for image_id, url in IMAGES.items():
        filepath = FIXTURES_DIR / f"{image_id}.jpg"
        
        if not filepath.exists():
            print(f"Downloading {image_id}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
        else:
            print(f"{image_id} already exists.")
            
        rows.append({
            "image_id": image_id,
            "modality": "general",
            "image_path": str(filepath.resolve())
        })
        
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "modality", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Manifest written to {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
