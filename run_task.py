import sys
import os
from PIL import Image

try:
    from uacloak.interface import generate_cloak, compare_faces
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)


def process_image(img_path):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return
    print(f"\nProcessing: {img_path}")
    image = Image.open(img_path).convert("RGB")

    generator = generate_cloak(image, epsilon=0.03, num_steps=100, alpha_fraction=0.1)
    final_output = None
    try:
        for output in generator:
            final_output = output
            # If output is a tuple, let's print it to see the structure
            # Let's assume (orig_image, cloaked_image, scores...)
    except Exception as e:
        print(f"Error during generate_cloak: {e}")
        return

    print(f"DEBUG: yielded type: {type(final_output)}")
    print(f"DEBUG: yielded value: {final_output}")

    if final_output:
        # Fallback to index-based access if it's a tuple
        try:
            # Based on previous AttributeError, it is a tuple
            # Expecting structure based on typical implementations:
            # (orig_img, cloaked_img, orig_score, cloak_score, status_text)
            # or similar.
            orig_img, cloaked_img, orig_score, cloak_score, status_text = final_output[
                :5
            ]
            print(f"Orig Score: {orig_score}")
            print(f"Cloak Score: {cloak_score}")
            print(f"Status: {status_text}")
            sim, summary = compare_faces(orig_img, cloaked_img)
            print(f"Comparison Similarity: {sim}")
            print(f"Comparison Summary: {summary}")
        except Exception as e:
            print(f"Error processing tuple: {e}")


process_image("tests/fixtures/faces/obama_a.jpg")
process_image("tests/fixtures/faces/obama_b.jpg")
