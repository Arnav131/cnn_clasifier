# Instructions: Local Machine Training and Evaluation Guide (CNN + MOGEO + Feature Fusion)

This document is the complete, step-by-step reference for training, optimizing, and evaluating the medicinal plant classification pipeline directly on your **local machine** using your **local GPU** (or CPU fallback).

The pipeline strictly implements the architecture defined in the project flowchart:

```
[Start]
   │
   ▼
[Image Collection] (dataset/ with 150 classes)
   │
   ▼
[Dataset Preparation] (build_manifest: 85% dev / 15% final_test, duplicate leakage fix)
   │
   ▼
[Image Preprocessing] (denoising, bilateral filtering, resizing, color normalization)
   │
   ▼
[Image Segmentation] (HSV masking + Otsu + morphological opening/closing + GrabCut refinement)
   │
   ▼
[Feature Extraction] (Shape, Size, Colour, Vein, Texture -> extracted_features.csv)
   │
   ▼
[MOGEO Optimization] (Feature Selection and CNN Hyperparameter Optimization) ◄──┐
   │                                                                            │
   ▼                                                                            │
[CNN Layers] (Hybrid Model Training: Conv visual features fused with features)   │ (No)
   │                                                                            │
   ▼                                                                            │
[Performance Evaluation] (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)│
   │                                                                            │
   ▼                                                                            │
< Model Performance Satisfactory? > ────────────────────────────────────────────┘
   │ (Yes)
   ▼
[Predict Plant Species] (Inference on new leaf image with Top-K confidence)
   │
   ▼
[Associated Medicinal Properties] (Lookup botanical name, family, properties, side effects)
   │
   ▼
[End]
```

---

## 1. System Requirements & Hardware Setup

### Recommended Hardware:
- **Operating System:** Windows 10 / 11 (64-bit), Linux, or macOS.
- **Python:** Python 3.10 or 3.11.
- **GPU (Recommended):** NVIDIA GPU with CUDA support (e.g., RTX 3050/3060/4060 or higher, 4GB+ VRAM).
- **RAM:** 8GB minimum (16GB recommended).
- **Disk Space:** ~5GB free space.

*(Note: If no NVIDIA GPU is present, the pipeline automatically detects CPU and runs with all features intact).*

---

## 2. Environment Setup (One-Time Setup)

Open **PowerShell** or **Command Prompt** in the project directory (`d:\ML_PROJECTS\code\code`):

### Step 2.1: Verify Python Installation
```powershell
python --version
```
Ensure it returns Python 3.10.x or higher.

### Step 2.2: (Optional but Recommended) Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run this once:
#  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)
```

### Step 2.3: Install PyTorch with CUDA for Local GPU Acceleration
If you have an **NVIDIA GPU**, install the CUDA-enabled build of PyTorch:
```powershell
# For CUDA 12.1 (recommended for modern drivers):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8 (for older cards/drivers):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If you do **not** have an NVIDIA GPU (running on CPU):
```powershell
pip install torch torchvision
```

### Step 2.4: Install Required Packages
```powershell
pip install -r requirements.txt
```

### Step 2.5: Verify GPU Availability
Run this quick check to confirm PyTorch detects your GPU:
```powershell
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Local)')"
```
You should see `CUDA Available: True` and your GPU's name printed.

---

## 3. Flowchart Step 1 & 2: Image Collection & Dataset Verification

Verify that your dataset directory exists and contains all 150 plant species folders:
```powershell
python -c "import os; classes = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]; print(f'Total plant classes found: {len(classes)}')"
```
Output should state: `Total plant classes found: 150`.

---

## 4. Flowchart Step 3: Dataset Preparation & Leakage-Free Splitting

Run the manifest builder to partition the dataset into stratified development and test splits:
```powershell
python -c "from pipeline.data import build_manifest; from pipeline.utils import leakage_report; df = build_manifest('dataset', 'splits'); print(leakage_report(df.to_dict('records'))); print(df['split'].value_counts())"
```

### What this does:
1. Performs **one stratified split**:
   - `final_test`: 15% held-out test split (strictly untouched until final evaluation).
   - Development set: 85% remaining images.
2. Stratifies development set into `dev_train` (~68%) and `dev_val` (~17%).
3. Performs byte-level MD5 hashing to eliminate duplicate image leakage across splits.
4. Saves `splits/manifest.csv` and `splits/classes.json`.

---

## 5. Flowchart Steps 4, 5 & 6: Image Preprocessing, Leaf Segmentation & Feature Extraction

Execute the preprocessing, segmentation, and feature extraction pipeline:
```powershell
python run_preprocess.py --dataset-dir dataset --output-dir .
```

### What this does:
1. **Image Preprocessing:** Cleans and normalizes images using bilateral filtering to remove noise while preserving leaf edge boundaries.
2. **Image Segmentation:** Isolates the leaf foreground from the background using Excess Green Index ($2G - R - B$), HSV green masking, morphological opening/closing, and GrabCut contour refinement.
   - Saves clean, segmented images to `preprocessed_dataset/`.
3. **Feature Extraction (36 Hand-Crafted Quantitative Features):**
   - **Shape (7 features):** Contour Area, Perimeter, Circularity ($4\pi A/P^2$), Aspect Ratio, Solidity, Extent, Eccentricity.
   - **Size (4 features):** Bounding Box Width, Height, Leaf Area Ratio, Equivalent Diameter.
   - **Colour (12 features):** Means and standard deviations of Red, Green, Blue, Hue, Saturation, and Value within the leaf mask.
   - **Vein (7 features):** Leaf interior Canny edge density, multi-orientation Gabor filter responses ($0^\circ, 45^\circ, 90^\circ, 135^\circ$), and Morphological Top-Hat/Black-Hat vein ridge responses.
   - **Texture (6 features):** Gray-Level Co-occurrence Matrix (GLCM) descriptors: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, Angular Second Moment (ASM).
4. Saves all features to `extracted_features.csv`.

*(Note: If you have already generated `extracted_features.csv` and want to skip re-extraction, pass `--skip-existing-csv`).*

---

## 6. Flowchart Step 7: Baseline Model Training (Fixed Architecture + AdamW)

Train the baseline CNN model to establish an honest benchmark before running metaheuristic search:
```powershell
python run_baseline.py --dataset-dir dataset --features-csv extracted_features.csv --epochs 20
```

### What this does:
- Trains a standard 3-block CNN using **AdamW** optimizer with feature fusion.
- Validates each epoch on `dev_val` (internal validation set) with early stopping.
- Saves checkpoints to `checkpoints/baseline_best.pt`.
- Saves training curves to `results/baseline_training_curves.png`.
- Logs run metrics to `results/experiment_log.csv`.

*(If interrupted, add `--resume` to continue training seamlessly).*

---

## 7. Flowchart Step 8: MOGEO Optimization (Feature Selection & CNN Hyperparameter Search)

Execute the Multi-Objective Golden Eagle Optimizer (MOGEO) to find the Pareto-optimal feature subsets and CNN architecture hyperparameters:
```powershell
python run_mogeo.py --dataset-dir dataset --features-csv extracted_features.csv --pop-size 8 --generations 6 --resume
```

### How MOGEO Works:
- **Feature Selection:** MOGEO optimizes binary inclusion flags for each of the 5 feature categories: Shape, Size, Colour, Vein, Texture, and the feature projection dimension (16, 32, 64).
- **Architecture & Training Optimization:** Simultaneously searches over CNN depth (`num_blocks`), channel capacity (`base_channels`), `dropout`, `lr`, `weight_decay`, `batch_size`, and `label_smoothing`.
- **Dual Objectives (both maximized on dev_val):**
  1. Internal Validation Accuracy
  2. Internal Validation Macro-F1
- **Flight Mechanics:** Golden Eagles navigate the hyperparameter space using attack flight vectors (toward Pareto archive prey) and cruise flight vectors (orthogonal Gram-Schmidt circling).
- **Resumability:** Checkpoints search state after *every single candidate evaluation* to `checkpoints/mogeo_state.pkl`. Re-running with `--resume` resumes instantly without losing progress.

### Outputs:
- `results/mogeo_results.csv`: Candidate evaluation metrics.
- `results/best_hparams.json`: The selected best compromise solution from the Pareto front.
- `results/mogeo_convergence.png`: Optimization convergence curves across generations.

---

## 8. Flowchart Step 9: Final CNN Model Training

Retrain the CNN using the best hyperparameters and feature selections discovered by MOGEO:
```powershell
python run_final.py --dataset-dir dataset --features-csv extracted_features.csv --epochs 40 --resume
```

### What this does:
- Loads the best compromise configuration from `results/best_hparams.json`.
- Trains the hybrid CNN (Conv visual layers + MOGEO-selected feature projection) on `dev_train` with AdamW.
- Validates on `dev_val` with early stopping (patience = 7).
- Generates **the final deployable model**: `checkpoints/best.pt`.
- Saves loss/accuracy curves to `results/training_curves.png`.

---

## 9. Flowchart Step 10: Performance Evaluation & Satisfaction Decision

Run the final, unbiased evaluation on the **untouched 15% final_test split**:
```powershell
python evaluate_final.py --dataset-dir dataset --features-csv extracted_features.csv
```

### Outputs Generated in `results/`:
- `results/accuracy_report.md`: Complete summary with Top-1 Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1.
- `results/accuracy_report.json`: Machine-readable metrics.
- `results/confusion_matrix.png`: High-resolution confusion matrix across all 150 species.
- `results/error_analysis.md`: Detailed failure analysis showing lowest-recall classes and top misclassification pairs with filenames.

### Flowchart Decision: "Model Performance Satisfactory?"
- **If NO:** Return to **Section 7 (MOGEO Optimization)**. Increase `--pop-size` or `--generations` to explore a wider search space.
- **If YES:** Proceed to **Section 10 (Predict Plant Species & Medicinal Properties)**!

---

## 10. Flowchart Steps 11 & 12: Predict Plant Species & Associated Medicinal Properties

Use the trained `checkpoints/best.pt` model to classify any leaf image and automatically look up its medicinal properties:

### Option A: Classify a Single Image
```powershell
python test_model.py --checkpoint checkpoints/best.pt --image "Amaranthus viridis (1).jpg" --properties-csv properties.csv
```

**Example Output:**
```text
Amaranthus viridis (1).jpg
  #1: Amaranthus viridis             94.8%
  #2: Spinacia oleracea               2.3%
  #3: Chenopodium album               1.1%
  -> Botanical name: Amaranthus viridis L.
  -> Family: Amaranthaceae
  -> Medicinal property: Anti-inflammatory, antioxidant, diuretic, vermifuge
  -> Side effects: Excessive consumption may lead to kidney stone formation due to oxalates
```

### Option B: Classify an Entire Folder of Images
```powershell
python test_model.py --checkpoint checkpoints/best.pt --dir path/to/unseen_leaves/ --properties-csv properties.csv
```

### Option C: Quick Sanity Verification on Final Test Split
```powershell
python test_model.py --checkpoint checkpoints/best.pt --eval-test --dataset-dir dataset --properties-csv properties.csv
```

---

## 11. Interactive Jupyter Notebook (Alternative)

If you prefer running interactively cell-by-cell in VS Code or JupyterLab instead of the terminal, open `Local_Training.ipynb` in your editor. It follows the exact same numbered steps above.

---

## 12. Troubleshooting & FAQ

- **Q: How do I ensure training uses my NVIDIA GPU?**
  Run `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, reinstall PyTorch using the CUDA index URL: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall`.
- **Q: Out of Memory (OOM) on GPU?**
  In `run_baseline.py`, `run_mogeo.py`, or `run_final.py`, reduce the batch size (e.g. `--batch-size 16`), or let MOGEO explore smaller batch sizes automatically.
- **Q: Can I run this without `extracted_features.csv`?**
  Yes. If `extracted_features.csv` is not present, all scripts automatically train with pure visual CNN representations. Generating `extracted_features.csv` with `run_preprocess.py` enables the hybrid feature fusion architecture described in the flowchart.
- **Q: Did the old fake numbers from `MainCode.ipynb` get removed?**
  Yes. All metrics produced by this pipeline are calculated from real forward passes over authentic data, logged in `results/experiment_log.csv` and `results/accuracy_report.md`.
