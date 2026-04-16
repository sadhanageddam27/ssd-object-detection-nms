# SSD Object Detection, NMS, and HOI Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Gemini](https://img.shields.io/badge/Google-Gemini%20API-blueviolet?logo=google)

Three-part Computer Vision project covering lightweight object detection
from scratch, custom NMS implementation verified against PyTorch, and
zero-shot Human-Object Interaction analysis using Vision-Language Models.

## Live Project Report
Full results with images and analysis:
**https://sadhanageddam27.github.io/project3/**

---

## Project Structure

```
ssd-object-detection-nms/
├── code/
│   ├── part1_LOD.ipynb       # Lightweight SSD object detector
│   ├── part2_NMS.ipynb       # Custom NMS vs torchvision.ops.nms
│   └── Part3_HOI.ipynb       # Zero-shot HOI using Google Gemini
└── report/
    └── Part3_full_Report.pdf
```

---

## What This Project Covers

### Part 1 — Lightweight SSD Object Detector
Built a Single Shot Detector (SSD) from scratch on the D2L Banana Dataset
(1000 train / 100 validation images).

**Architecture:**
- Lightweight convolutional backbone
- Two detection feature maps: 32×32 and 16×16
- Multi-scale anchors — 8 per location
- Dual-head output: classification + bounding box regression

**Training:**
- Loss = Cross-Entropy (classification) + Smooth L1 (bbox regression)
- Hard-negative mining at 3:1 ratio
- Adam optimizer with step learning-rate scheduler over 20 epochs

**Results:**
- Loss curves decrease smoothly — no overfitting
- Correct detections on validation set across varied backgrounds
- Tested on real-world banana images not in the training set

| Case | Outcome |
|------|---------|
| Simple background, normal shape | Correct detection, tight bbox |
| Banana in fruit basket | Detected with reasonable confidence |
| Heavy clutter / occlusion | Missed — distribution mismatch |
| Peeled banana | Failed — shape too far from training data |

### Part 2 — Custom NMS Implementation
Implemented Non-Maximum Suppression from scratch and verified against
`torchvision.ops.nms`.

**Algorithm:**
1. Sort all predicted boxes by confidence score
2. Select highest-scoring box
3. Compute IoU with all remaining boxes
4. Suppress boxes with IoU > threshold (default 0.5)

**Verification result:**
```
My NMS indices:          tensor([14])
Torchvision NMS indices: tensor([14])
Same set of indices?     True
```
Custom implementation matches PyTorch exactly across IoU thresholds
of 0.3, 0.5, and 0.7.

### Part 3 — Human-Object Interaction Using VLMs
Zero-shot HOI detection on 5 HICO-DET images using Google Gemini,
comparing model predictions against ground-truth HOI labels.

**Setup:**
- Dataset: HuggingFace HICO-DET (`zhimeng/hico_det`)
- Model: Google Gemini (zero-shot, no HICO-DET fine-tuning)
- Prompt format: `List all interactions as <verb object>`

**Key findings:**

| Image | Ground Truth | Gemini Result |
|-------|-------------|---------------|
| Soccer | `sports_ball kick` | `kick ball` — semantic match, ontology mismatch |
| Wine glass | `wine_glass hold` | `hold glass` — same issue |
| Cake | `cake cut` | `cut cake` — correct with extra hallucinations |
| Truck | `truck no_interaction` | Missed entirely — VLMs struggle with negation |
| Boat | `boat ride/sit_on` | Partially correct + hallucinated motorcycle |

**Failure modes identified:**
- Ontology mismatch: Gemini uses natural language, HICO-DET uses dataset-specific names
- Clothing hallucinations: `wear shirt`, `wear pants` appear across all images
- Negation blindspot: `no_interaction` labels never predicted correctly

**Prompt refinement** (added verb list + object list + 3-shot examples) improved
boat and cake scenes but did not resolve clothing hallucinations or negation cases.

---

## Setup and Usage

```bash
# Clone the repo
git clone https://github.com/sadhanageddam27/ssd-object-detection-nms.git
cd ssd-object-detection-nms

# Install dependencies
pip install torch torchvision matplotlib jupyter datasets google-generativeai

# Open notebooks
jupyter notebook code/part1_LOD.ipynb
jupyter notebook code/part2_NMS.ipynb
jupyter notebook code/Part3_HOI.ipynb
```

For Part 3 (HOI), you will need a Google Gemini API key set as an
environment variable: `GEMINI_API_KEY=your_key_here`

---

## Tech Stack
Python · PyTorch · torchvision · Jupyter · Google Gemini API · HuggingFace Datasets · Matplotlib

## Topics
`computer-vision` `object-detection` `ssd` `non-maximum-suppression` `pytorch`
`vision-language-models` `human-object-interaction` `deep-learning` `python`
