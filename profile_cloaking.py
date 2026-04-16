import time
from pathlib import Path
from PIL import Image

from facecloak.pipeline import detect_primary_face, extract_embedding_numpy
from facecloak.cloaking import cloak_face_tensor, CloakHyperparameters
from facecloak.models import get_face_detector, get_embedding_model


def main():
    print("Initializing models...")
    t0 = time.time()
    detector = get_face_detector()
    model = get_embedding_model()
    print(f"Models initialized in {time.time() - t0:.2f}s")

    img = Image.open(Path("tests/fixtures/faces/obama_a.jpg")).convert("RGB")

    print("MTCNN detection...")
    t0 = time.time()
    detected = detect_primary_face(img)
    print(f"Detection took {time.time() - t0:.2f}s")

    print("Extracting embedding...")
    t0 = time.time()
    emb = extract_embedding_numpy(detected.tensor)
    print(f"Embedding extraction took {time.time() - t0:.2f}s")

    print("Running PGD loop (100 steps)...")
    t0 = time.time()
    result = cloak_face_tensor(
        detected.tensor, parameters=CloakHyperparameters(num_steps=100)
    )
    print(f"PGD loop took {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
