import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# FaceCloak Technical Walkthrough\nThis notebook walks through a single complete cloaking operation, exposing the mathematics and intermediate tensors step by step."),
    nbf.v4.new_code_cell("import torch\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom PIL import Image\nfrom pathlib import Path\nfrom facecloak.models import get_face_detector, get_embedding_model\nfrom facecloak.pipeline import detect_primary_face, extract_embedding_numpy, cosine_similarity\nfrom facecloak.cloaking import cloak_face_tensor, CloakHyperparameters"),
    nbf.v4.new_markdown_cell("## 1. Load Original Image & Face Detection"),
    nbf.v4.new_code_cell("img = Image.open(Path('tests/fixtures/faces/obama_a.jpg')).convert('RGB')\ndetected = detect_primary_face(img)\nplt.imshow(detected.image)\nplt.title('MTCNN Aligned Face (160x160)')\nplt.axis('off')\nplt.show()"),
    nbf.v4.new_markdown_cell("## 2. Baseline Embedding\nThe model (InceptionResnetV1) embeds this face into a 512-dimensional vector. We will track how this embedding moves during the attack."),
    nbf.v4.new_code_cell("orig_embedding = extract_embedding_numpy(detected.tensor)\nprint(f'Original embedding shape: {orig_embedding.shape}')\nprint(f'Original embedding L2 norm: {np.linalg.norm(orig_embedding):.4f}')"),
    nbf.v4.new_markdown_cell("## 3. Projected Gradient Descent\nWe run 100 steps of L-infinity PGD with epsilon=0.03. We capture the perturbation at multiple steps."),
    nbf.v4.new_code_cell("result = cloak_face_tensor(detected.tensor, parameters=CloakHyperparameters(epsilon=0.03, num_steps=100))"),
    nbf.v4.new_markdown_cell("## 4. Visualizing Adversarial Noise\nTo humans, the modification is completely invisible. The amplitude is tiny (<= 0.03 normalized). We amplify it by 75x to visualize its structure."),
    nbf.v4.new_code_cell("plt.imshow(result.amplified_diff)\nplt.title('Amplified Adversarial Noise (75x)')\nplt.axis('off')\nplt.show()"),
    nbf.v4.new_markdown_cell("## 5. Resulting Cloak\nThe final cloaked embedding is drastically far in cosine distance from the original."),
    nbf.v4.new_code_cell("print(f'Original Vs Original Similarity: {cosine_similarity(orig_embedding, orig_embedding):.4f}')\nprint(f'Original Vs Cloaked Similarity: {result.final_similarity:.4f}')\n\nfig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))\nax1.imshow(detected.image)\nax1.set_title('Original')\nax1.axis('off')\nax2.imshow(result.cloaked_face_image)\nax2.set_title('Cloaked')\nax2.axis('off')\nplt.show()")
]

with open('technical_writeup.ipynb', 'w') as f:
    nbf.write(nb, f)
