

## 🧠 Sabse pehle: kya karna hai?

Tere project ka **simple flow** bas itna hai:

```text
DATASET
   ↓
DATASET SPLIT
   ↓
PREPROCESSING + SEGMENTATION
   ↓
FEATURE EXTRACTION
   ↓
BASELINE CNN
   ↓
MOGEO OPTIMIZATION
   ↓
FINAL CNN
   ↓
FINAL TEST
   ↓
PREDICTION
```

**Bas.** 😭
Abhi Golden Eagle ke equations, Pareto front, Gabor, GLCM etc. mein mat ghus.

---

# 🚀 TERA ACTUAL STEP-BY-STEP PLAN

Main tujhe **ekdum execution order** mein bata raha hoon.

### STEP 0 — Project folder open kar

Tera terminal project ke root folder mein hona chahiye:

```text
d:\ML_PROJECTS\code\code
```

Yaani jahan ye files/folders hain:

```text
code/
├── dataset/
├── pipeline/
├── run_preprocess.py
├── run_baseline.py
├── run_mogeo.py
├── run_final.py
├── evaluate_final.py
├── test_model.py
├── requirements.txt
└── ...
```

---

# STEP 1 — Environment ready karo

PowerShell:

```powershell
python --version
```

Python **3.10/3.11** hona chahiye. Guide bhi isi range ko recommend karta hai. 

Phir:

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

Phir packages:

```powershell
pip install -r requirements.txt
```

---

# STEP 2 — GPU check 🔥

Ye **bahut important** hai.

Run:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Agar output:

```text
True
NVIDIA GeForce RTX ...
```

toh **perfect**.

Agar:

```text
False
CPU
```

toh rukna — pehle GPU/PyTorch setup fix karenge.

Guide bhi CUDA availability verify karne ko bolta hai. 

---

# STEP 3 — Dataset check 🌿

Tere `dataset/` folder mein **150 plant classes** honi chahiye.

Run:

```powershell
python -c "import os; classes=[d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset',d))]; print('Classes:',len(classes))"
```

Expected:

```text
Classes: 150
```

Agar **150 nahi aata**, yahin stop kar aur mujhe output bhej.

---

# STEP 4 — Dataset split

Ab ye command:

```powershell
python -c "from pipeline.data import build_manifest; from pipeline.utils import leakage_report; df=build_manifest('dataset','splits'); print(leakage_report(df.to_dict('records'))); print(df['split'].value_counts())"
```

Iska kaam simple hai:

```text
Dataset
   │
   ├── dev_train
   ├── dev_val
   │
   └── final_test   ← FINAL TEST, untouched
```

Guide ke according:

* 85% development
* 15% final test
* development ke andar train/validation split
* duplicate images ko leakage se remove/check kiya jata hai. 

### ⚠️ Important

`final_test` ko **MOGEO ke time use nahi karna**.

Warna final result biased ho sakta hai.

---

# STEP 5 — Preprocessing + Segmentation + Features

Ab:

```powershell
python run_preprocess.py --dataset-dir dataset --output-dir .
```

Ye ek command **3 major kaam** karegi:

### 1️⃣ Image preprocessing

```text
Original image
      ↓
Noise removal
      ↓
Filtering
      ↓
Normalization
```

### 2️⃣ Leaf segmentation

```text
Image
 ↓
Leaf detect
 ↓
Background remove
 ↓
Clean leaf
```

### 3️⃣ Feature extraction

36 handcrafted features:

```text
Shape
Size
Colour
Vein
Texture
```

Aur final file banegi:

```text
extracted_features.csv
```

Ye sab guide ke preprocessing section mein defined hai. 

---

# STEP 6 — Baseline CNN 🧪

**MOGEO immediately mat chala dena.**

Pehle normal CNN ka benchmark chahiye.

Run:

```powershell
python run_baseline.py --dataset-dir dataset --features-csv extracted_features.csv --epochs 20
```

Ye basically answer karega:

> **"MOGEO ke bina normal CNN kitna perform karta hai?"**

Output:

```text
checkpoints/
    baseline_best.pt

results/
    baseline_training_curves.png
    experiment_log.csv
```

Guide specifically baseline ko MOGEO se pehle benchmark ke liye use karta hai. 

---

# STEP 7 — 🔥 MOGEO

**Ye sabse expensive step hai.**

Ab:

```powershell
python run_mogeo.py \
    --dataset-dir dataset \
    --features-csv extracted_features.csv \
    --pop-size 8 \
    --generations 6 \
    --resume
```

Windows PowerShell mein agar `\` issue kare toh **single line** mein:

```powershell
python run_mogeo.py --dataset-dir dataset --features-csv extracted_features.csv --pop-size 8 --generations 6 --resume
```

---

## MOGEO actually kar kya raha hai?

Isko complicated naam se confuse mat hona.

MOGEO basically ye decide karne ki koshish kar raha hai:

```text
        MOGEO
          │
     ┌────┴────┐
     ↓         ↓
Features     CNN settings
select       optimize
     │         │
     ↓         ↓
Shape?       CNN depth?
Size?        Channels?
Colour?      Dropout?
Vein?        Learning rate?
Texture?     Batch size?
```

Guide ke according MOGEO feature categories aur CNN hyperparameters simultaneously search karta hai, while optimizing validation accuracy + macro-F1. 

### Output:

```text
results/
├── mogeo_results.csv
├── best_hparams.json
└── mogeo_convergence.png
```

**Sabse important file:**

```text
results/best_hparams.json
```

Ye MOGEO ka selected configuration hai.

---

# STEP 8 — Final CNN

Ab MOGEO se jo configuration mili hai uske basis par final model train hoga.

Run:

```powershell
python run_final.py --dataset-dir dataset --features-csv extracted_features.csv --epochs 40 --resume
```

Ye banayega:

```text
checkpoints/
    best.pt
```

**`best.pt` = tera final trained model.** 🔥

Guide ke according final training MOGEO ke selected configuration ko load karke dev_train par train karta hai aur dev_val par early stopping karta hai. 

---

# STEP 9 — FINAL EVALUATION 📊

Ab finally:

```powershell
python evaluate_final.py --dataset-dir dataset --features-csv extracted_features.csv
```

**Yahi tera actual final evaluation hai.**

Ye `final_test` par evaluate karega.

Outputs:

```text
results/
├── accuracy_report.md
├── accuracy_report.json
├── confusion_matrix.png
└── error_analysis.md
```

Metrics:

```text
Accuracy
Precision
Recall
Macro-F1
Weighted-F1
```

Guide explicitly says final evaluation untouched 15% `final_test` par hota hai. 

---

# STEP 10 — Model ko actual image pe test karo 🌿

Agar final model ban gaya:

```text
checkpoints/best.pt
```

Then:

```powershell
python test_model.py --checkpoint checkpoints/best.pt --image "Amaranthus viridis (1).jpg" --properties-csv properties.csv
```

Output something like:

```text
Prediction:

#1 Amaranthus viridis     94.8%
#2 Spinacia oleracea       2.3%
#3 Chenopodium album       1.1%

Botanical name: ...
Family: ...
Medicinal property: ...
Side effects: ...
```

Guide mein isi type ka prediction flow diya hai. 

---

# 🧩 Ekdum simple mental model

Bhai is project ko **6 boxes** mein yaad rakh:

```text
┌─────────────────┐
│ 1. DATASET      │
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. PREPROCESS   │
│ + FEATURES      │
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. BASELINE CNN │
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. MOGEO        │
│ optimize        │
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. FINAL CNN    │
└────────┬────────┘
         ↓
┌─────────────────┐
│ 6. TEST +       │
│ PREDICTION      │
└─────────────────┘
```

### Tujhe abhi **ye nahi karna**:

❌ MOGEO theory ratna
❌ Golden Eagle equations samajhna
❌ Pareto mathematics mein jaana
❌ Har feature manually calculate karna
❌ 10 scripts ek saath run karna
❌ Final test baar-baar use karna

### Tujhe abhi **sirf ye karna hai**:

```text
1. GPU CHECK
      ↓
2. 150 CLASSES CHECK
      ↓
3. BUILD SPLIT
      ↓
4. RUN PREPROCESS
      ↓
5. RUN BASELINE
      ↓
6. RUN MOGEO
      ↓
7. RUN FINAL
      ↓
8. EVALUATE
      ↓
9. TEST IMAGE
```

---

## 🔥 Aur ek important baat

**Tu saari commands ek saath mat chala.**

Hum **checkpoint-by-checkpoint** jayenge.

### Abhi sirf ye 3 commands chala:

```powershell
python --version
```

then

```powershell
python -c "import torch; print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

then

```powershell
python -c "import os; classes=[d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset',d))]; print('Classes:',len(classes))"
```

**Bas inka output mujhe bhej de.**
Uske baad main tujhe **exactly next command** dunga — aur hum project ko ek-ek checkpoint pe complete karenge. 💪
