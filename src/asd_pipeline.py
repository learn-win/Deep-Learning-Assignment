"""DA-ArcNet pipeline as a plain script. Author: Ardhanarieswarar (24I461). Team: Gokul Kannan (24I463), Jafer Sathik B (24I464), Madhan M (24I466). PSG College of Technology, Department of Information Technology.
GENERATED from notebook/ASD_24I461_Ardhanarieswarar.ipynb - edit the notebook, not this file.
Cells are separated by `# %%` markers. Written for Google Colab (TensorFlow >= 2.16)."""

# %% [markdown]
# # DA-ArcNet: Lightweight Domain-Adversarial Angular-Margin Learning for Unsupervised Anomalous Sound Detection of Industrial Machines under Domain Shift
# 
# **Author:** 24I461 Ardhanarieswarar  |  **Team members:** 24I463 Gokul Kannan  |  24I464 Jafer Sathik B  |  24I466 Madhan M
# 
# **Institution:** PSG College of Technology, Department of Information Technology  
# **Course:** 23I006 Deep Learning Mini Project  |  **Framework:** TensorFlow / Keras  |  **Runtime:** Google Colab (T4 GPU)
# 
# ---
# ### What this notebook contains (mapped to the evaluation rubric)
# 
# | Rubric item | Where in this notebook |
# |---|---|
# | 1. Problem formulation and literature survey | Section 0 (problem, 18 verified 2022-26 papers, research gap, objectives) |
# | 2. Dataset handling and modality-specific pre-processing (both datasets) | Sections 2-5b: MIMII DG + ToyADMOS2 (ToyCar/ToyTrain), EDA, log-mel pipeline, leakage-free splits, raw-vs-processed input figures |
# | 3. Model design and novelty | Section 6: proposed **DA-ArcNet** (adaptive clip-mean normalisation + gradient-reversal domain adversary + angular-margin embedding + kNN scoring + AE fusion), architecture diagram, parameter count |
# | 4. Experimentation and evaluation | Sections 9-21: 3 baselines (DCASE22 AE, DCASE22 MobileNetV2, ArcFace-kNN) + proposed model, all task metrics, 95 % bootstrap CIs and Wilcoxon tests (10), noise robustness (16), ablation (17), cross-dataset evaluation (18), multi-seed runs (19), model size (21) |
# | 5. Result analysis, explainability, discussion | Section 20 (AE residual maps, Grad-CAM, t-SNE + domain probe, error-case analysis), Section 22 (limitations and future work), Section 27 |
# | 6. Conference-ready paper | Section 23 exports LaTeX tables, figures and auto-written findings for the IEEE paper in the repo (`paper/`) |
# | 7. Presentation, reproducibility, viva | Section 24 builds the slide deck from your real results; Section 25 live demo (Gradio); the repository has README, requirements, `docs/viva_prep.md` |
# 
# ### How to run
# 1. **Runtime -> Change runtime type -> T4 GPU.**
# 2. Set `QUICK_TEST = True` in the config cell and run all (about 15-20 min): this checks the whole pipeline on one machine. Then set it back to `False`.
# 3. **Runtime -> Run all** for the full study. The full run downloads about 6 GB and takes roughly **3-4 hours** on a T4 (multi-seed and cross-dataset parts can be shortened in the config).
# 4. Everything (figures, tables, models, paper assets, slides) is saved to **Google Drive -> `ASD_project/`**. If Colab disconnects, run again: finished machines are skipped (`RESUME = True`).
# 
# ### Notes
# * The raw ToyADMOS2 record (zenodo 4580270) is 169 GB and needs a separate mixer tool, so it cannot be used directly on Colab. The **DCASE 2022 development set** (zenodo 6355122) contains the ToyCar and ToyTrain machines of ToyADMOS2, already split into train/test with source/target domains and identical in format to MIMII DG; those two machines are the ToyADMOS2 part of this study.
# * TensorFlow does not use the PyTorch `.pt` format. Models are saved as `.keras`, `.weights.h5` and SavedModel folders.
# * This notebook has not been executed on the real data by its author-assistant; all numbers in your report must come from **your** Colab run.

# %% [markdown]
# ## 0. Problem formulation and literature survey
# 
# ### 0.1 Problem
# Factories monitor machines by sound: a change in the sound of a fan, valve or bearing can signal a fault before a breakdown. In practice only **normal** sounds are available for training, because faults are rare and varied. The task is therefore **unsupervised anomalous sound detection (ASD)**: learn from normal clips only, then decide for a new 10-second clip whether the machine is normal or anomalous.
# 
# The difficulty is **domain shift**. Operating speed, load, microphone position and background noise change between the training (*source*) and deployment (*target*) conditions, and these changes can alter a clip's sound more than a real fault does. A model that only learned the source condition raises false alarms on shifted but healthy machines (or misses faults). The DCASE 2022 benchmark formalises this: each machine section has many source-domain and only a few target-domain normal training clips, and the test set mixes source and target clips with no domain label given to the model [1].
# 
# **Formally.** For a machine section, given normal training clips $\mathcal{D}=\{x_i\}$ drawn mostly from the source domain and a handful from the target domain, learn an anomaly score $s(x)$ such that $s(x_{\text{anomalous}}) > s(x_{\text{normal}})$ in *both* domains, and choose a threshold without using any anomalous data.
# 
# ### 0.2 Literature survey (verified, 2022-2026)
# All entries were checked against their arXiv abstract pages. Descriptions restate only what the title/abstract says.
# 
# | # | Authors | Year | Title | What it contributes | Limitation relevant to this work |
# |---|---|---|---|---|---|
# | 1 | Dohi et al. | 2022 | Description and Discussion on DCASE 2022 Challenge Task 2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring Applying Domain Generalization Techniques (arXiv:2206.05876) | Defines the domain-generalisation ASD benchmark (source/target domains, AUC and pAUC) and the AE and MobileNetV2 baselines used in this work. | Benchmark paper; baselines do not address domain shift explicitly. |
# | 2 | Dohi et al. | 2022 | MIMII DG: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection for Domain Generalization Task (arXiv:2205.13879) | MIMII DG dataset: five machine types, three sections per type, each section a different kind of domain shift. | Dataset paper. |
# | 3 | Liu et al. | 2022 | Anomalous Sound Detection using Spectral-Temporal Information Fusion (arXiv:2201.05510) | STgram-MFN: fuses spectrogram and temporal features for self-supervised machine-ID classification. | Uses only the in-domain machine ID task; no explicit domain-invariance. |
# | 4 | Zeng et al. | 2023 | Joint Generative-Contrastive Representation Learning for Anomalous Sound Detection (arXiv:2305.12111) | GeCo: self-attention predictive autoencoder plus multi-task contrastive objective; evaluated on DCASE 2020 Task 2. | Evaluated on a single benchmark year without domain shift. |
# | 5 | Zhang et al. | 2023 | Anomalous Sound Detection Using Self-Attention-Based Frequency Pattern Analysis of Machine Sounds (arXiv:2308.14063) | Self-attention over frequency patterns, combined with spectral and temporal information; DCASE 2020 Task 2. | No domain-shift setting evaluated. |
# | 6 | Dohi et al. | 2023 | Description and Discussion on DCASE 2023 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring (arXiv:2305.07828) | Extends the benchmark to the first-shot setting. | Benchmark paper. |
# | 7 | Wilkinghoff | 2023 | Self-Supervised Learning for Anomalous Sound Detection (arXiv:2312.09578) | Proposes feature exchange (FeatEx) as a self-supervised approach and reports state-of-the-art results on DCASE 2023. | Ensembles of auxiliary-task embeddings; no cross-dataset transfer study. |
# | 8 | Wilkinghoff | 2024 | AdaProj: Adaptively Scaled Angular Margin Subspace Projections for Anomalous Sound Detection with Auxiliary Classification Tasks (arXiv:2403.14179) | AdaProj loss: projects embeddings onto class-specific subspaces with an angular margin; evaluated on DCASE 2022 and 2023. | Embedding-loss improvement; domain shift handled only through data and scoring. |
# | 9 | Nishida et al. | 2024 | Description and Discussion on DCASE 2024 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring (arXiv:2406.07250) | DCASE 2024 first-shot task description. | Benchmark paper. |
# | 10 | Jiang et al. | 2024 | AnoPatch: Towards Better Consistency in Machine Anomalous Sound Detection (arXiv:2406.11364) | Addresses consistency between pre-training and ASD fine-tuning (as stated in the title). | Relies on large pre-trained models. |
# | 11 | Saengthong and Shinozaki | 2024 | Deep Generic Representations for Domain-Generalized Anomalous Sound Detection (arXiv:2409.05035) | GenRep: generic features from a large pre-trained extractor with kNN scoring, MemMixup and domain normalisation; Official Score 73.79% on the DCASE 2023 Task 2 evaluation set. | Large pre-trained extractor; heavy for edge deployment. |
# | 12 | Zheng et al. | 2024 | Improving Anomalous Sound Detection via Low-Rank Adaptation Fine-Tuning of Pre-Trained Audio Models (arXiv:2409.07016) | LoRA fine-tuning of pre-trained audio models for ASD. | Large pre-trained models; high compute. |
# | 13 | Purohit et al. | 2024 | MIMII-Gen: Generative Modeling Approach for Simulated Evaluation of Anomalous Sound Detection System (arXiv:2409.18542) | Generative simulation of data for evaluating ASD systems (as stated in the title). | Evaluation methodology rather than detection model. |
# | 14 | Meng et al. | 2024 | Contrastive Learning with Spectrum Information Augmentation in Abnormal Sound Detection (arXiv:2509.15570) | Contrastive learning with spectrum-level augmentation for abnormal sound detection. | Augmentation-based; no explicit domain-adversarial component. |
# | 15 | Wilkinghoff et al. | 2025 | Handling Domain Shifts for Anomalous Sound Detection: A Review of DCASE-Related Work (arXiv:2503.10435) | Reviews domain generalisation work for ASD from the DCASE challenges; domain shifts such as microphone type or placement can affect audio more than anomalies do. | Review; motivates robust, domain-aware design. |
# | 16 | Nishida et al. | 2025 | Description and Discussion on DCASE 2025 Challenge Task 2: First-shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring (arXiv:2506.10097) | DCASE 2025 first-shot task description. | Benchmark paper. |
# | 17 | Han et al. | 2025 | Exploring Self-Supervised Audio Models for Generalized Anomalous Sound Detection (arXiv:2508.12230) | Fine-tunes speech/audio self-supervised models with LoRA, a Machine-aware Group Adapter and a dual-level contrastive loss; evaluated on DCASE 2020-2024 datasets. | Large pre-trained backbones; no lightweight design. |
# | 18 | Gao et al. | 2026 | UD-ASD: A Unified Diffusion Model for Anomalous Sound Detection (arXiv:2607.12576) | Single diffusion model on log-mel spectrograms with machine-ID condition embeddings and GMM scoring; improves over the baseline by 3.44% AUC and 2.52% pAUC on DCASE 2022 Task 2. | Generative model; domain shift not handled explicitly. |
# 
# ### 0.3 Research gap
# Based on the papers above:
# 1. **Heavy models.** Recent high-scoring systems rely on large pre-trained backbones, LoRA fine-tuning or ensembles [11, 12, 17], which are costly for edge / factory deployment. A *lightweight* model with a strong domain-shift handling mechanism is less explored.
# 2. **Implicit domain handling.** Several works improve the embedding loss [8] or use memory banks and normalisation [11], but do not train the embedding to be *explicitly invariant* to the source/target domain while keeping an angular-margin condition classifier.
# 3. **Single-benchmark evaluation.** The surveyed methods are evaluated within one challenge dataset at a time; **cross-dataset transfer** (e.g. MIMII DG <-> ToyADMOS2) and robustness to additional noise are rarely reported alongside the main score.
# 4. **Limited analysis.** The abstracts focus on AUC/pAUC; explainability (where in time-frequency a decision comes from) and error-case analysis receive little attention.
# 
# ### 0.4 Objectives
# * **O1.** Design a lightweight ASD model that is explicitly robust to domain shift (**DA-ArcNet**).
# * **O2.** Compare it with three baselines (DCASE22 AE, DCASE22 MobileNetV2, ArcFace-kNN in the style of recent top systems) on MIMII DG and ToyADMOS2 using the official AUC / pAUC protocol.
# * **O3.** Test **cross-dataset** transfer in both directions (MIMII DG -> ToyADMOS2 and back).
# * **O4.** Quantify each design component with an **ablation**, and robustness with additive-noise tests.
# * **O5.** Provide statistical evidence (bootstrap CIs, multiple seeds, Wilcoxon tests) and **explainability** (Grad-CAM, AE residual maps, t-SNE).

# %% [markdown]
# ## 1. Setup and configuration
# All hyper-parameters are in one dictionary (`CFG`). Change values here only.

# %%
import os, sys, re, json, math, time, random, shutil, hashlib, warnings, zipfile, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy import signal as sps
from scipy.stats import gamma, hmean
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix, matthews_corrcoef)
from sklearn.decomposition import PCA

try:
    from tqdm.auto import tqdm
except Exception:
    class tqdm:                      # minimal stand-in if tqdm is missing
        def __init__(self, x=None, **kw): self.x = x
        def __iter__(self): return iter(self.x)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, n=1): pass
try:
    from IPython.display import display
except Exception:
    display = print

import tensorflow as tf

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})

IN_COLAB = "google.colab" in sys.modules

# ----------------------------- CONFIG -----------------------------
CFG = dict(
    SEED=42,
    QUICK_TEST=False,            # True = 1 machine, 3 epochs: a fast sanity check of the whole pipeline
    MIMII_MACHINES=["fan", "gearbox", "bearing", "slider", "valve"],
    TOY_MACHINES=["ToyCar", "ToyTrain"],
    RUN_MACHINES=None,           # None = all seven machines; or e.g. ["fan", "valve"]
    # audio / features
    SR=16000, N_FFT=1024, HOP=512, N_MELS=64,
    # baseline 1: autoencoder (DCASE 2022 baseline style)
    AE_FRAMES=5, AE_TRAIN_STRIDE=3, AE_LATENT=8, AE_EPOCHS=60, AE_BATCH=512, AE_LR=1e-3, AE_PATIENCE=8,
    # crop-based models (MobileNetV2 baseline, ArcFace-kNN baseline, proposed DA-ArcNet)
    CLF_CROP=64, CLF_INFER_STRIDE=32, CLF_CROPS_PER_CLIP=4, CLF_EPOCHS=30, CLF_BATCH=64, CLF_LR=1e-3,
    EMB_DIM=128, ARC_S=30.0, ARC_M=0.3, KNN_K=3,
    DANN_LAMBDA=0.3,             # max strength of the gradient-reversal domain adversary (ramped up during training)
    TARGET_OVERSAMPLE=10,        # each target-domain training clip is seen this many times per epoch (only 10 clips / section)
    AUG=True,
    # protocol
    VAL_FRAC=0.10,               # share of SOURCE normal training clips held out (thresholds, early stopping, score scaling)
    THR_PERCENTILE=0.90,         # decision threshold = 90th percentile of a gamma fitted to held-out normal scores
    MAX_FPR=0.1,                 # pAUC range
    NORM_MODE="global",          # "global" (per-mel mean/std) or "cmn" (clip mean normalisation)
    # robustness study
    NOISE_SNRS=[20, 10, 5, 0], NOISE_KINDS=["white", "pink"],
    # statistics and extra experiments
    BOOT_N=500,                  # bootstrap resamples for 95 % confidence intervals
    ABLATION_MACHINES=["fan", "ToyCar"],   # machines on which the two extra ablation variants are trained ([] to skip)
    RUN_SEEDS=True, SEED_MACHINES=["fan", "ToyCar"], EXTRA_SEEDS=[1, 2],      # multi-seed repeats (mean +/- std)
    RUN_CROSS_DATASET=True, XD_EPOCHS=20,                                      # MIMII DG <-> ToyADMOS2 transfer
    # housekeeping
    RESUME=True, USE_DRIVE=True, DELETE_AUDIO_AFTER=True,
    DEBUG=True,                  # True = print debug lines in every cell (banners, shapes, value ranges, warnings)
)
if CFG["QUICK_TEST"]:
    CFG.update(RUN_MACHINES=["fan"], AE_EPOCHS=3, CLF_EPOCHS=3, NOISE_SNRS=[10], NOISE_KINDS=["white"],
               ABLATION_MACHINES=["fan"], RUN_SEEDS=False, RUN_CROSS_DATASET=False, BOOT_N=50, RESUME=False)
if CFG["RUN_MACHINES"] is None:
    CFG["RUN_MACHINES"] = CFG["MIMII_MACHINES"] + CFG["TOY_MACHINES"]

# ----------------------------- PATHS ------------------------------
WORK = Path("/content/asd_work") if IN_COLAB else Path("./asd_work")
OUT_DIR = WORK / "outputs"
if IN_COLAB and CFG["USE_DRIVE"]:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT_DIR = Path("/content/drive/MyDrive/ASD_project")
    except Exception as e:
        print("Google Drive not mounted, saving to local disk instead:", e)
if CFG["QUICK_TEST"]:
    OUT_DIR = OUT_DIR / "quick_test"
DATA_DIR = WORK / "data"
FEAT_DIR = (OUT_DIR.parent / "ASD_cache") if (IN_COLAB and CFG["USE_DRIVE"] and OUT_DIR != WORK / "outputs") else WORK / "features"
FIG_DIR, TAB_DIR, MODEL_DIR = OUT_DIR / "figures", OUT_DIR / "tables", OUT_DIR / "models"
for d in (DATA_DIR, FEAT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("Outputs ->", OUT_DIR)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    tf.keras.utils.set_random_seed(s)


set_seed(CFG["SEED"])


def savefig(name):
    plt.savefig(FIG_DIR / f"{name}.png")
    plt.show()
    dbg(f"figure saved -> {FIG_DIR / (name + '.png')}")


# ----------------------------- DEBUG HELPERS -----------------------------
CELL_LOG = []                    # (cell, seconds, new variables) for every cell that ran; printed in the summary cell


def dbg(msg):
    # one debug line (switch off with CFG["DEBUG"] = False)
    if CFG["DEBUG"]:
        print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def describe(name, a, warn=True):
    # Print shape / dtype / min / max / mean / std / NaN / Inf of an array and warn about problems. Returns a dict.
    a = np.asarray(a)
    fin = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, bool)
    n_nan, n_inf = int(np.isnan(a).sum()) if a.dtype.kind == "f" else 0, int(np.isinf(a).sum()) if a.dtype.kind == "f" else 0
    v = a[fin] if a.size else a
    st = dict(name=name, shape=tuple(a.shape), dtype=str(a.dtype), min=float(v.min()) if v.size else np.nan,
              max=float(v.max()) if v.size else np.nan, mean=float(v.mean()) if v.size else np.nan,
              std=float(v.std()) if v.size else np.nan, nan=n_nan, inf=n_inf)
    if CFG["DEBUG"]:
        print(f"[DEBUG] {name:<34} shape={str(st['shape']):<18} {st['dtype']:<8} min={st['min']:>9.3f} max={st['max']:>9.3f} "
              f"mean={st['mean']:>8.3f} std={st['std']:>8.3f} NaN={n_nan} Inf={n_inf}")
        if warn and (n_nan or n_inf):
            print(f"[DEBUG] WARNING: {name} contains {n_nan} NaN and {n_inf} Inf values!")
        if warn and v.size and st["std"] < 1e-9:
            print(f"[DEBUG] WARNING: {name} is constant (std = 0)")
    return st


def _brief(x):
    if isinstance(x, np.ndarray):
        return f"array{tuple(x.shape)} {x.dtype}"
    if isinstance(x, pd.DataFrame):
        return f"DataFrame{x.shape}"
    if isinstance(x, (list, tuple, set, dict)):
        return f"{type(x).__name__}[{len(x)}]"
    if isinstance(x, (int, float, str, bool, Path)):
        return f"{type(x).__name__}={str(x)[:40]}"
    return type(x).__name__


def dbg_start(cell):
    if not CFG["DEBUG"]:
        return None
    print(f"\n[DEBUG {time.strftime('%H:%M:%S')}] ===== cell '{cell}' START =====", flush=True)
    return time.time(), set(globals().keys())


def dbg_end(cell, tok):
    if tok is None:
        return
    t0, before = tok
    sec = time.time() - t0
    new = [k for k in globals().keys() if k not in before and not k.startswith("_")]
    CELL_LOG.append(dict(cell=cell, seconds=round(sec, 1), new_variables=", ".join(new[:10]) + (" ..." if len(new) > 10 else "")))
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] ===== cell '{cell}' DONE in {sec:.1f} s =====", flush=True)
    if new:
        print("[DEBUG] new variables: " + " | ".join(f"{k}: {_brief(globals()[k])}" for k in new[:8]) + (" ..." if len(new) > 8 else ""))


ALL_MODELS = ["ae", "mnv2", "arc", "arc_ada", "arc_dann", "prop", "full"]
MODELS = ["ae", "mnv2", "arc", "prop", "full"]            # main comparison
MODEL_NAMES = {"ae": "AE baseline [DCASE22]", "mnv2": "MobileNetV2 [DCASE22]", "arc": "ArcFace-kNN [SOTA-style]",
               "arc_ada": "ArcFace-kNN + AdaCMN", "arc_dann": "ArcFace-kNN + DANN",
               "prop": "Proposed DA-ArcNet", "full": "Proposed + AE fusion"}
MODEL_COLORS = {"ae": "#7f7f7f", "mnv2": "#bcbd22", "arc": "#1f77b4", "arc_ada": "#17becf", "arc_dann": "#9467bd",
                "prop": "#d62728", "full": "#2ca02c"}

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__, "| GPUs:", gpus)
if not gpus:
    print("WARNING: no GPU found. Use Runtime -> Change runtime type -> T4 GPU (training will be very slow on CPU).")
print("Machines to run:", CFG["RUN_MACHINES"])
dbg(f"setup finished | DEBUG={CFG['DEBUG']} | seed={CFG['SEED']} | quick_test={CFG['QUICK_TEST']} | out_dir={OUT_DIR}")

# %% [markdown]
# ## 2. Data download (Zenodo)
# * **MIMII DG** machines come from zenodo record **6529888** (the link from the project sheet).
# * **ToyCar / ToyTrain** (ToyADMOS2) come from the DCASE 2022 development set, zenodo record **6355122**.
# 
# Every file is verified with its MD5 checksum from the Zenodo page. Each machine has 3 *sections* (each section is one kind of domain shift); per section: 990 normal **source** + 10 normal **target** clips for training, and 50 normal + 50 anomalous clips for each domain for testing. Anomalous sounds never appear in training.

# %%
_dbg_tok = dbg_start("download")          # --- debug (auto): banner + timer ---
DATASETS = {   # machine: (url, md5)
    "bearing":  ("https://zenodo.org/records/6529888/files/bearing.zip?download=1",  "6381a00f9efc0ced779c8ad847e4ff59"),
    "fan":      ("https://zenodo.org/records/6529888/files/fan.zip?download=1",      "a1a9b488934a82426bacc933d87aacde"),
    "gearbox":  ("https://zenodo.org/records/6529888/files/gearbox.zip?download=1",  "c165dfef8c404256bd719c6fe1f7036f"),
    "slider":   ("https://zenodo.org/records/6529888/files/slider.zip?download=1",   "8c3a5466cf53e54872fd94998a67bfac"),
    "valve":    ("https://zenodo.org/records/6529888/files/valve.zip?download=1",    "1da37b2e82942dfba720984541e2ef60"),
    "ToyCar":   ("https://zenodo.org/records/6355122/files/dev_ToyCar.zip?download=1",   "1ccca43751353e516ee7d6d55457c705"),
    "ToyTrain": ("https://zenodo.org/records/6355122/files/dev_ToyTrain.zip?download=1", "17e5b8cae15ccb630a5c533551bc93b5"),
}


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def has_audio(machine):
    d = DATA_DIR / machine
    return d.exists() and any(d.rglob("section_*_test_*.wav"))


def fetch(url, dest, tries=6):
    # Resumable download with a visible progress bar (works in Colab; re-run the cell to continue after a break).
    import requests
    dest = Path(dest)
    for attempt in range(1, tries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if r.status_code == 416:                       # already complete
                    return
                r.raise_for_status()
                resumed = (r.status_code == 206)
                total = int(r.headers.get("content-length", 0)) + (have if resumed else 0)
                with open(dest, "ab" if resumed else "wb") as f, \
                        tqdm(total=total or None, initial=have if resumed else 0, unit="B", unit_scale=True,
                             unit_divisor=1024, desc=dest.name, mininterval=2) as bar:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            return
        except Exception as e:
            print(f"   download interrupted ({type(e).__name__}: {str(e)[:80]}) - retry {attempt}/{tries} in 10 s")
            time.sleep(10)
    raise RuntimeError(f"Could not download {url}")


def download_machine(machine, tries=2):
    # Download + verify + unzip one machine (skipped if audio is already on disk).
    if has_audio(machine):
        print(f"[{machine}] audio already on disk")
        return
    url, md5 = DATASETS[machine]
    zp = DATA_DIR / f"{machine}.zip"
    for attempt in range(1, tries + 1):
        print(f"[{machine}] downloading from Zenodo (attempt {attempt}) - several GB, this takes a few minutes; the bar below shows progress")
        fetch(url, zp)
        print(f"[{machine}] checking MD5 ...")
        if md5sum(zp) == md5:
            print(f"[{machine}] checksum OK")
            break
        print(f"[{machine}] checksum mismatch - downloading again")
        zp.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"Could not download a valid copy of {machine}.zip")
    out = DATA_DIR / machine
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{machine}] extracting ...")
    with zipfile.ZipFile(zp) as z:
        z.extractall(out)
    zp.unlink(missing_ok=True)
    print(f"[{machine}] extracted to {out}")
dbg_end("download", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 3. Metadata parsing and log-mel features
# File names carry all labels: `section_00_source_train_normal_0000_<attributes>.wav` -> section, domain (source/target), split, normal/anomaly and the operating-condition attributes. Log-mel spectrograms are computed once per machine with plain NumPy and cached.

# %%
_dbg_tok = dbg_start("features")          # --- debug (auto): banner + timer ---
FN_RE = re.compile(r"section_(\d+)_(source|target)_(train|test)_(normal|anomaly)_(\d+)_?(.*)\.wav$")


def parse_name(fname):
    m = FN_RE.match(Path(fname).name)
    if not m:
        return None
    sec, dom, split, cond, idx, attrs = m.groups()
    attrs = attrs.strip("_") or "noAttribute"
    return dict(section=int(sec), domain=dom, split=split, label=int(cond == "anomaly"),
                idx=int(idx), attrs=attrs, cls_key=f"{int(sec)}|{attrs}")


def parse_meta(machine):
    rows = []
    for p in sorted((DATA_DIR / machine).rglob("*.wav")):
        d = parse_name(p.name)
        if d is None:
            continue
        d.update(path=str(p), name=p.name)
        rows.append(d)
    meta = pd.DataFrame(rows)
    if meta.empty:
        raise RuntimeError(f"No dataset wav files found for {machine} under {DATA_DIR / machine}")
    return meta.reset_index(drop=True)


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    h2m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    m2h = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    hz = m2h(np.linspace(h2m(fmin), h2m(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((bins - l) / (c - l), (r - bins) / (r - c)))
    fb *= (2.0 / (hz[2:n_mels + 2] - hz[:n_mels]))[:, None]
    return fb


MEL_FB = mel_filterbank(CFG["SR"], CFG["N_FFT"], CFG["N_MELS"])
HANN = np.hanning(CFG["N_FFT"] + 1)[:-1].astype(np.float32)


def read_wave(path):
    sr, w = wavfile.read(path)
    if w.ndim > 1:
        w = w[:, 0]
    if w.dtype == np.int16:
        w = w.astype(np.float32) / 32768.0
    elif w.dtype == np.int32:
        w = w.astype(np.float32) / 2147483648.0
    else:
        w = w.astype(np.float32)
    if sr != CFG["SR"]:
        w = sps.resample_poly(w, CFG["SR"], sr).astype(np.float32)
    return w


def logmel(wave):
    # waveform -> log-mel spectrogram, shape [frames, n_mels] (dB)
    n_fft, hop = CFG["N_FFT"], CFG["HOP"]
    if len(wave) < n_fft:
        wave = np.pad(wave, (0, n_fft - len(wave)))
    frames = np.lib.stride_tricks.sliding_window_view(wave, n_fft)[::hop]
    power = np.abs(np.fft.rfft(frames * HANN, axis=1)) ** 2
    mel = power @ MEL_FB.T
    return (10.0 * np.log10(mel + 1e-10)).astype(np.float32)


def feat_paths(machine):
    tag = f"{CFG['N_MELS']}m_{CFG['N_FFT']}f_{CFG['HOP']}h"
    return FEAT_DIR / f"{machine}_{tag}_logmel.npy", FEAT_DIR / f"{machine}_{tag}_meta.csv"


def get_data(machine):
    # Returns (meta DataFrame, log-mel array [N, T, n_mels]); cached on local disk.
    npy, csv = feat_paths(machine)
    if npy.exists() and csv.exists():
        meta = pd.read_csv(csv, dtype={"attrs": str, "cls_key": str, "name": str, "path": str})
        X = np.load(npy)
        dbg(f"get_data[{machine}]: loaded cached features {X.shape} | clips {len(meta)} | "
            f"train {int((meta.split == 'train').sum())} / test {int((meta.split == 'test').sum())} | NaN {int(np.isnan(X).sum())}")
        return meta, X
    dbg(f"get_data[{machine}]: no cache -> download + feature extraction")
    download_machine(machine)
    meta = parse_meta(machine)
    dbg(f"get_data[{machine}]: found {len(meta)} wav files; sections {sorted(meta.section.unique().tolist())}")
    with ThreadPoolExecutor(4) as ex:
        feats = list(tqdm(ex.map(lambda p: logmel(read_wave(p)), meta.path), total=len(meta),
                          desc=f"log-mel [{machine}]"))
    T = min(f.shape[0] for f in feats)
    if len(set(f.shape[0] for f in feats)) > 1:
        dbg(f"get_data[{machine}]: clips differ in length ({min(f.shape[0] for f in feats)}..{max(f.shape[0] for f in feats)} frames) -> cut to {T}")
    X = np.stack([f[:T] for f in feats]).astype(np.float32)
    describe(f"{machine} log-mel (raw features)", X)
    np.save(npy, X)
    meta.to_csv(csv, index=False)
    return meta, X
dbg_end("features", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 4. Exploratory analysis: what does the domain shift look like?
# Class counts, example spectrograms (source vs target, normal vs anomaly), the average spectrum per domain, and a PCA view of clip-level features.

# %%
_dbg_tok = dbg_start("eda")          # --- debug (auto): banner + timer ---
eda_machine = CFG["RUN_MACHINES"][0]
meta_e, X_e = get_data(eda_machine)
describe(f"{eda_machine} log-mel (EDA)", X_e)
dbg(f"clips per (section, domain, split, label) are shown in the table below; expected 990 source + 10 target train clips per section")
print(f"Machine: {eda_machine} | clips: {len(meta_e)} | log-mel shape per clip: {X_e.shape[1:]}")
display(meta_e.groupby(["section", "domain", "split", "label"]).size()
        .unstack(["split", "label"]).fillna(0).astype(int))

# example spectrograms of section 0
sec0 = meta_e[meta_e.section == 0]
picks = [("source", "train", 0), ("target", "train", 0), ("source", "test", 1), ("target", "test", 1)]
titles = ["source - normal (train)", "target - normal (train)", "source - ANOMALY (test)", "target - ANOMALY (test)"]
fig, axs = plt.subplots(1, 4, figsize=(18, 3.6))
for ax, (dom, split, lab), t in zip(axs, picks, titles):
    sel = sec0[(sec0.domain == dom) & (sec0.split == split) & (sec0.label == lab)]
    if len(sel) == 0:
        ax.axis("off"); continue
    im = ax.imshow(X_e[sel.index[0]].T, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(t); ax.set_xlabel("frame"); ax.set_ylabel("mel bin")
fig.suptitle(f"{eda_machine}: log-mel spectrograms (section 00)", y=1.03)
savefig("eda_spectrograms")

# average spectrum per group
fig, axs = plt.subplots(1, 3, figsize=(17, 3.8), sharey=True)
for s, ax in zip(sorted(meta_e.section.unique()), axs):
    g = meta_e[meta_e.section == s]
    for (dom, split, lab), col in zip([("source", "train", 0), ("target", "train", 0), ("source", "test", 1), ("target", "test", 1)],
                                      ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
        idx = g[(g.domain == dom) & (g.split == split) & (g.label == lab)].index
        if len(idx):
            ax.plot(X_e[idx].mean(axis=(0, 1)), color=col, label=f"{dom} {'anomaly' if lab else 'normal'} ({split})")
    ax.set_title(f"section {s:02d}"); ax.set_xlabel("mel bin")
axs[0].set_ylabel("mean log-mel (dB)"); axs[0].legend(fontsize=8)
fig.suptitle(f"{eda_machine}: average spectrum per domain", y=1.03)
savefig("eda_mean_spectrum")

# PCA of clip-level descriptors (mean + std over time)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
for s, ax in zip(sorted(meta_e.section.unique()), axs):
    g = meta_e[meta_e.section == s]
    D = np.concatenate([X_e[g.index].mean(axis=1), X_e[g.index].std(axis=1)], axis=1)
    D = (D - D.mean(0)) / (D.std(0) + 1e-6)
    Z = PCA(2, random_state=0).fit_transform(D)
    for (dom, lab), col, mk in zip([("source", 0), ("target", 0), ("source", 1), ("target", 1)],
                                   ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], ["o", "o", "x", "x"]):
        m = ((g.domain == dom) & (g.label == lab)).values
        if m.any():
            ax.scatter(Z[m, 0], Z[m, 1], s=14, c=col, marker=mk, alpha=0.6, label=f"{dom} {'anomaly' if lab else 'normal'}")
    ax.set_title(f"section {s:02d}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
axs[0].legend(fontsize=8)
fig.suptitle(f"{eda_machine}: PCA of clip features - source and target form separate clusters (domain shift)", y=1.03)
savefig("eda_pca_domains")
del meta_e, X_e
dbg_end("eda", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 5. Pre-processing helpers
# * **Normalisation** is fitted on training clips only (`global`: per-mel mean/std; `cmn`: subtract each clip's own per-mel mean first, which removes constant level / noise offsets).
# * **Autoencoder input** = 5 consecutive log-mel frames flattened (5 x 64 = 320 values).
# * **Classifier input** = random 64-frame crops (about 2 s) of the log-mel image, with light SpecAugment-style augmentation during training.

# %%
_dbg_tok = dbg_start("prep")          # --- debug (auto): banner + timer ---
def fit_norm(X, mode):
    # Fit normalisation statistics on training clips X [N, T, M].
    Z = X - X.mean(axis=1, keepdims=True) if mode == "cmn" else X
    mu = np.zeros(X.shape[2], np.float32) if mode == "cmn" else X.mean(axis=(0, 1)).astype(np.float32)
    sd = (Z - mu).std(axis=(0, 1)).astype(np.float32) + 1e-6
    return dict(mode=mode, mu=mu, sd=sd)


def apply_norm(X, st):
    if st["mode"] == "cmn":
        X = X - X.mean(axis=1, keepdims=True)
    return ((X - st["mu"]) / st["sd"]).astype(np.float32)


def make_windows(F, frames, stride=1):
    # [N, T, M] -> [N, W, frames*M] (stacked consecutive frames)
    N, T, M = F.shape
    starts = np.arange(0, T - frames + 1, stride)
    return np.stack([F[:, s:s + frames, :].reshape(N, -1) for s in starts], axis=1)


def random_crops(F, y, k, crop, rng, idx=None):
    # k random crops per clip. idx = clip indices to draw from (may repeat -> oversampling); y aligned with idx.
    idx = np.arange(len(F)) if idx is None else np.asarray(idx)
    N, T = len(idx), F.shape[1]
    starts = rng.integers(0, T - crop + 1, size=(N, k))
    tt = starts[:, :, None] + np.arange(crop)[None, None, :]
    X = F[idx[:, None, None], tt, :]
    return X.reshape(N * k, crop, F.shape[2]), np.repeat(y, k)


def fixed_crops(F, crop, stride):
    N, T, M = F.shape
    starts = np.arange(0, T - crop + 1, stride)
    return np.stack([F[:, s:s + crop, :] for s in starts], axis=1), len(starts)   # [N, k, crop, M]


def augment_crops(X, rng):
    # Light augmentation: noise, level shift, one frequency mask, one time mask. X: [B, C, M]
    B, C, M = X.shape
    X = X + rng.normal(0, 0.1, X.shape).astype(np.float32)
    X = X + rng.normal(0, 0.2, (B, 1, 1)).astype(np.float32)
    f, f0 = rng.integers(0, 9, B), rng.integers(0, M - 8, B)
    mf = (np.arange(M)[None, :] >= f0[:, None]) & (np.arange(M)[None, :] < (f0 + f)[:, None])
    X = np.where(mf[:, None, :], 0.0, X)
    t, t0 = rng.integers(0, 11, B), rng.integers(0, C - 10, B)
    mt = (np.arange(C)[None, :] >= t0[:, None]) & (np.arange(C)[None, :] < (t0 + t)[:, None])
    X = np.where(mt[:, :, None], 0.0, X)
    return X.astype(np.float32)


def add_noise(w, snr_db, kind, rng):
    # Add white or pink (1/f) noise at a given SNR (dB) relative to the clip power.
    n = rng.standard_normal(len(w)).astype(np.float32)
    if kind == "pink":
        Nf = np.fft.rfft(n)
        f = np.arange(len(Nf), dtype=np.float32)
        f[0] = 1.0
        n = np.fft.irfft(Nf / np.sqrt(f), n=len(w)).astype(np.float32)
    ps, pn = np.mean(w ** 2) + 1e-12, np.mean(n ** 2) + 1e-12
    return (w + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))).astype(np.float32)


def knn_scores(E_q, sec_q, E_ref, sec_ref, k=3):
    # Cosine-kNN anomaly score. E_*: L2-normalised crop embeddings [N, crops, D].
    # Per query clip: for each crop, 1 - mean of the k largest cosine similarities to the reference crops
    # of the SAME section (references = normal training clips); then mean over crops.
    out = np.zeros(len(E_q))
    D = E_q.shape[-1]
    for s in np.unique(sec_q):
        R = E_ref[sec_ref == s].reshape(-1, D)
        qi = np.where(sec_q == s)[0]
        Q = E_q[qi].reshape(-1, D)
        sim = Q @ R.T
        kk = min(k, sim.shape[1])
        top = -np.partition(-sim, kk - 1, axis=1)[:, :kk]
        out[qi] = (1.0 - top.mean(axis=1)).reshape(len(qi), -1).mean(axis=1)
    return out


def section_scores_from_probs(P, sections, eps=1e-8):
    # DCASE22 MobileNetV2-baseline score: mean over crops of -log p(true section). P: [N, crops, n_sections]
    out = np.zeros(len(P))
    for i, sec in enumerate(sections):
        out[i] = -np.log(np.clip(P[i][:, int(sec)], eps, 1.0)).mean()
    return out


def split_indices(meta, base_seed):
    # train (normal, minus held-out), held-out normal source clips (val), official test clips
    rng = np.random.default_rng(base_seed)
    meta["is_val"] = False
    for sec, g in meta[(meta.split == "train") & (meta.domain == "source")].groupby("section"):
        n = max(1, int(round(len(g) * CFG["VAL_FRAC"])))
        meta.loc[rng.choice(g.index.values, n, replace=False), "is_val"] = True
    tr = np.where((meta.split == "train") & (~meta.is_val))[0]
    va = np.where(meta.is_val)[0]
    te = np.where(meta.split == "test")[0]
    assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va)), "data leakage between splits"
    dbg(f"split_indices: train {len(tr)} | held-out {len(va)} | test {len(te)} | overlap train/test = {len(set(tr) & set(te))}, "
        f"held-out/test = {len(set(va) & set(te))}, train/held-out = {len(set(tr) & set(va))} (all must be 0)")
    return tr, va, te


# ---- debug self-test: run every helper on random data (catches shape / normalisation bugs before any training starts)
_r = np.random.default_rng(0)
_X = _r.normal(-30, 12, (6, 120, CFG["N_MELS"])).astype(np.float32)
_st = fit_norm(_X, "global"); _F = apply_norm(_X, _st)
assert abs(_F.mean()) < 0.05 and abs(_F.std() - 1) < 0.05, "global normalisation should give mean 0 / std 1"
_W = make_windows(_F, CFG["AE_FRAMES"], 1)
assert _W.shape == (6, 120 - CFG["AE_FRAMES"] + 1, CFG["AE_FRAMES"] * CFG["N_MELS"]), _W.shape
_C, _ = random_crops(_F, np.arange(6), 3, CFG["CLF_CROP"], _r)
assert _C.shape == (18, CFG["CLF_CROP"], CFG["N_MELS"]), _C.shape
_FC, _k = fixed_crops(_F, CFG["CLF_CROP"], 16)
assert _FC.shape == (6, _k, CFG["CLF_CROP"], CFG["N_MELS"]), _FC.shape
assert augment_crops(_C, _r).shape == _C.shape
_E = _r.normal(size=(4, 2, 8)).astype(np.float32); _E /= np.linalg.norm(_E, axis=-1, keepdims=True)
_ks = knn_scores(_E, np.array([0, 0, 1, 1]), _E, np.array([0, 0, 1, 1]), 1)
assert np.allclose(_ks, 0, atol=1e-5), "a clip compared with itself must have kNN distance 0"
dbg(f"prep self-test passed: normalisation mean/std = {_F.mean():.3f}/{_F.std():.3f} | AE windows {_W.shape} | CNN crops {_C.shape} | "
    f"fixed crops {_FC.shape} | kNN self-distance {float(_ks.max()):.1e}")
dbg_end("prep", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 5b. Raw input vs. processed input (what the model actually receives)
# For two example clips (a normal source clip and an anomalous target clip) this section shows every stage, from the **raw recording** to the **exact arrays fed to the models**:
# 
# 1. raw waveform -> 2. raw power spectrogram (STFT) -> 3. log-mel spectrogram -> 4. normalised log-mel (**model input**) -> 5a. autoencoder input (5-frame window) / 5b. classifier input (64-frame crop) / 5c. the crop with training augmentation -> 6. value distribution before vs. after normalisation.
# 
# Change `VIZ_MACHINE` to inspect another machine.

# %%
_dbg_tok = dbg_start("viz_input")          # --- debug (auto): banner + timer ---
VIZ_MACHINE = CFG["RUN_MACHINES"][0]     # change to any machine name to inspect another one
meta_v, X_v = get_data(VIZ_MACHINE)
download_machine(VIZ_MACHINE)             # raw audio is needed for the waveform plot (no-op if already on disk)
st_v = fit_norm(X_v[(meta_v.split == "train").values], CFG["NORM_MODE"])   # statistics from training clips only
F_v = apply_norm(X_v, st_v)
T_v = F_v.shape[1]
describe("viz: log-mel BEFORE normalisation", X_v)
describe("viz: log-mel AFTER normalisation", F_v)
hop_s = CFG["HOP"] / CFG["SR"]
D_ae = CFG["AE_FRAMES"] * CFG["N_MELS"]


def pick(dom, split, lab, sec=0):
    s = meta_v[(meta_v.section == sec) & (meta_v.domain == dom) & (meta_v.split == split) & (meta_v.label == lab)]
    return None if s.empty else s.index[0]


def stft_db(w):
    frames = np.lib.stride_tricks.sliding_window_view(w, CFG["N_FFT"])[::CFG["HOP"]]
    P = np.abs(np.fft.rfft(frames * HANN, axis=1)) ** 2
    return (10 * np.log10(P + 1e-10)).astype(np.float32)


examples = [("source NORMAL clip (train)", pick("source", "train", 0)),
            ("target ANOMALOUS clip (test)", pick("target", "test", 1))]
examples = [(t, i) for t, i in examples if i is not None]

for k, (title, i) in enumerate(examples):
    row = meta_v.loc[i]
    w = read_wave(row["path"])
    S = stft_db(w)
    lm, fn = X_v[i], F_v[i]
    t0a, t0c = min(100, T_v - CFG["AE_FRAMES"]), min(100, T_v - CFG["CLF_CROP"])
    win = make_windows(fn[None], CFG["AE_FRAMES"], 1)[0]                   # [n_windows, 320]  autoencoder input
    crop = fn[t0c:t0c + CFG["CLF_CROP"]]                                   # [64, 64]          classifier input
    crop_aug = augment_crops(crop[None], np.random.default_rng(0))[0]
    dur = len(w) / CFG["SR"]
    ext_mel = [0, lm.shape[0] * hop_s, 0, CFG["N_MELS"]]

    fig, axs = plt.subplots(4, 2, figsize=(15, 15.5))
    ax = axs[0, 0]
    ax.plot(np.arange(len(w)) / CFG["SR"], w, lw=0.4, color="#1f77b4")
    ax.set_xlim(0, dur); ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude")
    ax.set_title(f"1. RAW INPUT - waveform ({len(w)} samples @ {CFG['SR']} Hz)")
    ax = axs[0, 1]
    im = ax.imshow(S.T, origin="lower", aspect="auto", cmap="magma", extent=[0, dur, 0, CFG["SR"] / 2])
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)"); ax.set_title("2. Raw power spectrogram (STFT, dB)")
    fig.colorbar(im, ax=ax, label="dB")
    ax = axs[1, 0]
    im = ax.imshow(lm.T, origin="lower", aspect="auto", cmap="magma", extent=ext_mel)
    ax.set_xlabel("time (s)"); ax.set_ylabel("mel bin"); ax.set_title(f"3. Log-mel spectrogram - BEFORE normalisation {lm.shape}")
    fig.colorbar(im, ax=ax, label="dB")
    ax = axs[1, 1]
    im = ax.imshow(fn.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-4, vmax=4, extent=ext_mel)
    ax.set_xlabel("time (s)"); ax.set_ylabel("mel bin"); ax.set_title(f"4. AFTER normalisation ('{CFG['NORM_MODE']}') = MODEL INPUT {fn.shape}")
    fig.colorbar(im, ax=ax, label="z-score")
    ax = axs[2, 0]
    im = ax.imshow(win[t0a].reshape(CFG["AE_FRAMES"], CFG["N_MELS"]), aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
    ax.set_xlabel("mel bin"); ax.set_ylabel("frame in window")
    ax.set_title(f"5a. Autoencoder input: {CFG['AE_FRAMES']} frames x {CFG['N_MELS']} mel = {D_ae} values (one of {len(win)} windows)")
    fig.colorbar(im, ax=ax, label="z-score")
    ax = axs[2, 1]
    im = ax.imshow(crop.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
    ax.set_xlabel("frame"); ax.set_ylabel("mel bin"); ax.set_title(f"5b. Classifier input: {CFG['CLF_CROP']}-frame crop ({CFG['CLF_CROP']} x {CFG['N_MELS']} x 1)")
    fig.colorbar(im, ax=ax, label="z-score")
    ax = axs[3, 0]
    im = ax.imshow(crop_aug.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
    ax.set_xlabel("frame"); ax.set_ylabel("mel bin"); ax.set_title("5c. Same crop WITH training augmentation (noise, level shift, freq/time mask)")
    fig.colorbar(im, ax=ax, label="z-score")
    ax = axs[3, 1]
    ax.hist(lm.ravel(), bins=60, color="#1f77b4", alpha=0.6)
    ax.set_xlabel("raw log-mel value (dB)", color="#1f77b4"); ax.set_ylabel("count")
    ax2 = ax.twiny()
    ax2.hist(fn.ravel(), bins=60, color="#ff7f0e", alpha=0.5)
    ax2.set_xlabel("normalised value (z-score)", color="#ff7f0e")
    ax.set_title("6. Value distribution: before (blue) vs after (orange) normalisation", pad=34)
    fig.suptitle(f"{VIZ_MACHINE} - {title} - {row['name']}", y=1.0, fontsize=13)
    plt.tight_layout()
    savefig(f"input_pipeline_{k + 1}")

    rows = [("1. raw waveform", w.shape, w), ("2. raw power spectrogram (dB)", S.shape, S),
            ("3. log-mel (before norm.)", lm.shape, lm), ("4. log-mel (after norm.) = model input", fn.shape, fn),
            ("5a. autoencoder input (all windows)", win.shape, win), ("5b. classifier input (one crop)", crop[..., None].shape, crop)]
    tab = pd.DataFrame([dict(stage=s, shape=str(tuple(sh)), min=float(a.min()), max=float(a.max()), mean=float(a.mean()), std=float(a.std()))
                        for s, sh, a in rows])
    print(f"\nInput pipeline summary - {title}")
    display(tab.round(3))
    tab.to_csv(TAB_DIR / f"input_pipeline_summary_{k + 1}.csv", index=False)
dbg_end("viz_input", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 5c. Data inspector: BEFORE vs AFTER pre-processing, for the whole dataset and for any single clip
# How to read it:
# 
# * **Part A - dataset audit (all machines):** statistics of the *raw* log-mel features (BEFORE) and of the *normalised* features that are fed to the models (AFTER), plus NaN checks. After normalisation the training data must have mean ~ 0 and std ~ 1.
# * **Part B - one clip, in numbers and pictures:** the actual values of the raw waveform, of the raw log-mel (BEFORE) and of the normalised log-mel (AFTER) side by side, and the exact tensors each model receives.
# * **Part C - domain shift before/after:** the average spectrum of source and target training clips before and after normalisation (normalisation does **not** remove the domain shift - that is the problem the model has to handle).
# 
# Set `INSPECT_MACHINE` / `INSPECT_CLIP` below (file name of any clip, or `None` for an automatic example) and run the cell again.

# %%
_dbg_tok = dbg_start("inspector")          # --- debug (auto): banner + timer ---
INSPECT_MACHINE = CFG["RUN_MACHINES"][0]
INSPECT_CLIP = None        # e.g. "section_00_target_test_anomaly_0012_vel_40_loc_P3.wav"  (None = automatic example)
SHOW_FRAMES, SHOW_MELS = 6, 8     # size of the number tables

# ======================= Part A: dataset audit, BEFORE vs AFTER =======================
audit, mean_curves = [], {}
for mc in CFG["RUN_MACHINES"]:
    npy, csv = feat_paths(mc)
    if not (npy.exists() and csv.exists()):
        dbg(f"audit: [{mc}] not processed yet (no feature cache) -> skipped")
        continue
    Xm = np.load(npy, mmap_mode="r")
    mt = pd.read_csv(csv, usecols=["split", "domain", "label"])
    tr_i = np.where((mt.split == "train").values)[0]
    Xt = np.asarray(Xm[tr_i])
    st_a = fit_norm(Xt, CFG["NORM_MODE"]); Ft = apply_norm(Xt, st_a)
    audit.append(dict(machine=mc, clips=len(mt), train=len(tr_i), test=len(mt) - len(tr_i), frames=Xm.shape[1], mels=Xm.shape[2],
                      BEFORE_mean=Xt.mean(), BEFORE_std=Xt.std(), BEFORE_min=Xt.min(), BEFORE_max=Xt.max(),
                      AFTER_mean=Ft.mean(), AFTER_std=Ft.std(), AFTER_min=Ft.min(), AFTER_max=Ft.max(),
                      NaN=int(np.isnan(Xt).sum()), Inf=int(np.isinf(Xt).sum())))
    mean_curves[mc] = (Xt.mean(axis=(0, 1)), Ft.mean(axis=(0, 1)))
    del Xt, Ft
audit = pd.DataFrame(audit)
if len(audit):
    print(f"PART A - dataset audit (training clips; normalisation mode '{CFG['NORM_MODE']}', statistics from the training clips only)")
    display(audit.round(3))
    audit.to_csv(TAB_DIR / "data_audit_before_after.csv", index=False)
    bad = audit[(audit.AFTER_mean.abs() > 0.1) | ((audit.AFTER_std - 1).abs() > 0.2) | (audit.NaN > 0)]
    dbg("audit result: " + ("all machines OK (AFTER mean ~ 0, std ~ 1, no NaN)" if bad.empty else f"CHECK these machines: {bad.machine.tolist()}"))
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.6))
    for mc, (b, a) in mean_curves.items():
        axs[0].plot(b, label=mc); axs[1].plot(a, label=mc)
    axs[0].set_title("BEFORE: mean log-mel per mel band (dB)"); axs[1].set_title("AFTER normalisation: mean per mel band (z-score)")
    for ax in axs:
        ax.set_xlabel("mel bin")
    axs[0].legend(fontsize=7)
    plt.tight_layout()
    savefig("audit_before_after")
else:
    print("PART A skipped: no machine has been processed yet (run Section 4 first).")


# ======================= Part B: one clip, BEFORE vs AFTER =======================
def inspect_clip(machine, name=None, n_frames=SHOW_FRAMES, n_mels=SHOW_MELS):
    meta, X = get_data(machine)
    download_machine(machine)                                    # raw audio (no-op if on disk)
    tr_m = (meta.split == "train").values
    st = fit_norm(X[tr_m], CFG["NORM_MODE"]); F = apply_norm(X, st)
    if name is None:
        cand = meta[(meta.split == "train") & (meta.domain == "source") & (meta.label == 0)]
        i = int(cand.index[0])
    else:
        hit = meta.index[meta["name"] == name]
        if len(hit) == 0:
            print(f"Clip '{name}' not found in {machine}. Examples you can use:")
            print(meta["name"].sample(5, random_state=0).tolist())
            return None
        i = int(hit[0])
    row = meta.loc[i]
    path = Path(row["path"])
    if not path.exists():
        path = next((DATA_DIR / machine).rglob(row["name"]))
    w = read_wave(path)
    lm, fn = X[i], F[i]
    print("=" * 100)
    print(f"CLIP {row['name']}\n  machine={machine} | section={row['section']} | domain={row['domain']} | split={row['split']} | "
          f"label={'ANOMALY' if row['label'] else 'normal'} | attributes={row['attrs']} | class key={row['cls_key']}")
    print(f"  file: {path} | {len(w)} samples @ {CFG['SR']} Hz = {len(w) / CFG['SR']:.2f} s")
    print("=" * 100)

    print("\nBEFORE pre-processing 1/3 - raw waveform, first 8 samples and statistics")
    display(pd.DataFrame({"sample": range(8), "amplitude": w[:8]}).round(5).T)
    describe("raw waveform", w)

    print(f"\nBEFORE pre-processing 2/3 - raw log-mel [dB]: first {n_frames} frames x first {n_mels} mel bins (shape {lm.shape})")
    before = pd.DataFrame(lm[:n_frames, :n_mels], columns=[f"mel{j}" for j in range(n_mels)]).round(2)
    display(before)
    print(f"AFTER pre-processing 3/3 - normalised ('{CFG['NORM_MODE']}') = the numbers the models get (same frames / bins)")
    after = pd.DataFrame(fn[:n_frames, :n_mels], columns=[f"mel{j}" for j in range(n_mels)]).round(2)
    display(after)
    describe("log-mel BEFORE (this clip)", lm)
    describe("log-mel AFTER  (this clip)", fn)
    z = (lm.mean(axis=0) - st["mu"]) / st["sd"] if CFG["NORM_MODE"] == "global" else fn.mean(axis=0)

    # exact tensors fed to each model
    win = make_windows(fn[None], CFG["AE_FRAMES"], 1)[0]
    crops, k = fixed_crops(fn[None], CFG["CLF_CROP"], CFG["CLF_INFER_STRIDE"])
    crops = crops[0][..., None]
    rows = [("Autoencoder (baseline 1)", f"[{len(win)} windows, {win.shape[1]}]", win),
            ("MobileNetV2 / ArcFace-kNN / DA-ArcNet", f"[{k} crops, {CFG['CLF_CROP']}, {CFG['N_MELS']}, 1]", crops)]
    print("\nWHAT THE MODELS RECEIVE for this clip (after pre-processing, before the network):")
    tab = pd.DataFrame([dict(model=m, tensor_shape=sh, dtype=str(a.dtype), min=a.min(), max=a.max(), mean=a.mean(), std=a.std()) for m, sh, a in rows])
    display(tab.round(3))
    if CFG["NORM_MODE"] == "global" and (np.abs(fn).max() > 12):
        dbg("WARNING: normalised values beyond +-12 sigma - check this clip (silence / clipping / different recording?)")

    fig, axs = plt.subplots(2, 3, figsize=(17, 8))
    t_ax = np.arange(len(w)) / CFG["SR"]
    axs[0, 0].plot(t_ax, w, lw=0.4); axs[0, 0].set_title("BEFORE: raw waveform"); axs[0, 0].set_xlabel("time (s)"); axs[0, 0].set_ylabel("amplitude")
    im = axs[0, 1].imshow(lm.T, origin="lower", aspect="auto", cmap="magma"); axs[0, 1].set_title("BEFORE: log-mel [dB]")
    fig.colorbar(im, ax=axs[0, 1])
    im = axs[0, 2].imshow(fn.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-4, vmax=4); axs[0, 2].set_title("AFTER: normalised log-mel = model input")
    fig.colorbar(im, ax=axs[0, 2])
    axs[1, 0].plot(lm.mean(axis=0), label="this clip (dB)"); axs[1, 0].plot(st["mu"] if CFG["NORM_MODE"] == "global" else lm.mean(axis=(0, 1)) * np.ones(lm.shape[1]),
                                                                             "--", label="training mean")
    axs[1, 0].set_title("BEFORE: mean spectrum of this clip vs training average"); axs[1, 0].set_xlabel("mel bin"); axs[1, 0].legend(fontsize=8)
    axs[1, 1].plot(fn.mean(axis=0)); axs[1, 1].axhline(0, color="k", lw=0.6)
    axs[1, 1].set_title("AFTER: mean per mel band (z-score; 0 = typical training clip)"); axs[1, 1].set_xlabel("mel bin")
    axs[1, 2].hist(lm.ravel(), bins=60, alpha=0.5, label="BEFORE (dB)", color="#1f77b4")
    ax2 = axs[1, 2].twiny(); ax2.hist(fn.ravel(), bins=60, alpha=0.5, label="AFTER (z)", color="#ff7f0e")
    axs[1, 2].set_title("value distribution: BEFORE (blue) vs AFTER (orange)", pad=30)
    fig.suptitle(f"{machine} | {row['name']}", y=1.0)
    plt.tight_layout()
    savefig(f"inspect_{machine}_{Path(row['name']).stem[:40]}")
    return dict(raw_waveform=w, logmel_before=lm, logmel_after=fn, ae_windows=win, cnn_crops=crops, row=row)


inspected = inspect_clip(INSPECT_MACHINE, INSPECT_CLIP)


# ======================= Part C: domain shift, BEFORE vs AFTER normalisation =======================
def domain_before_after(machine, section=0):
    meta, X = get_data(machine)
    tr_m = (meta.split == "train").values
    st = fit_norm(X[tr_m], CFG["NORM_MODE"]); F = apply_norm(X, st)
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.6))
    for dom, col in (("source", "#1f77b4"), ("target", "#ff7f0e")):
        idx = meta.index[(meta.section == section) & (meta.domain == dom) & tr_m]
        axs[0].plot(X[idx].mean(axis=(0, 1)), color=col, label=f"{dom} ({len(idx)} clips)")
        axs[1].plot(F[idx].mean(axis=(0, 1)), color=col, label=f"{dom} ({len(idx)} clips)")
    axs[0].set_title(f"{machine} section {section}: BEFORE normalisation (dB)"); axs[1].set_title("AFTER normalisation (z-score)")
    for ax in axs:
        ax.set_xlabel("mel bin"); ax.legend(fontsize=8)
    plt.tight_layout()
    savefig(f"domain_before_after_{machine}")
    dbg(f"{machine}: the source and target curves still differ after normalisation -> the domain shift is NOT removed by pre-processing")


domain_before_after(INSPECT_MACHINE)
dbg_end("inspector", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 6. Models (TensorFlow / Keras)
# 
# Five scores are produced for every test clip (all trained on **normal clips only**):
# 
# | Key | Model | Role |
# |---|---|---|
# | `ae` | Dense autoencoder on 5-frame log-mel windows, score = reconstruction error | **Baseline 1** - DCASE 2022 AE baseline style [1] |
# | `mnv2` | MobileNetV2 (alpha 0.5) trained to classify the *section*, score = -log p(section) | **Baseline 2** - DCASE 2022 MobileNetV2 baseline style [1], [MobileNetV2] |
# | `arc` | LiteCNN + ArcFace head over (section x operating-condition) classes, score = cosine-kNN distance to normal training embeddings | **Baseline 3** - re-implementation of the attribute-classification + angular-margin + kNN scoring paradigm of recent top systems [7, 8, 11] |
# | `prop` | **Proposed DA-ArcNet** = `arc` + **AdaCMN** + **gradient-reversal domain adversary** | **Proposed model** |
# | `full` | `prop` fused with `ae` (z-scores of log-scores on held-out normal clips) | Proposed system with reconstruction cue |
# 
# ### What is new in DA-ArcNet (design contributions)
# 1. **AdaCMN - adaptive clip-mean normalisation layer.** A learnable per-mel-band blend `x - sigmoid(a_f) * mean_t(x)` between "keep the level" and "remove the clip's own mean spectrum". Constant level, microphone and noise offsets are a main cause of source/target differences; the network learns, band by band, how much of them to remove (its contribution is isolated in the ablation by adding it alone to the ArcFace-kNN baseline).
# 2. **Domain-adversarial angular-margin embedding.** A gradient-reversal layer (strength ramped up during training) feeds a small domain classifier (source vs target), so the L2-normalised embedding is trained to be *uninformative about the domain* while an ArcFace head keeps it discriminative for the machine's operating condition [Ganin et al., Deng et al.]. Target clips (only 10 per section) are oversampled so the adversary sees both domains.
# 3. **Lightweight design.** Separable-convolution backbone (parameter count and latency are measured and reported in Section 21), kNN scoring with no attribute labels needed at test time.
# 4. **Hybrid score.** Optional z-score fusion with the autoencoder's reconstruction error (`full`).
# 
# *These are design contributions of this project built on known components; whether each helps is tested by the ablation (Section 17), not assumed.*

# %%
_dbg_tok = dbg_start("models")          # --- debug (auto): banner + timer ---
import contextlib, io
L = tf.keras.layers
register = tf.keras.utils.register_keras_serializable


# ------------------------- custom layers (serialisable) -------------------------
@register(package="asd")
class AdaCMN(L.Layer):
    # Learnable per-mel blend between identity and clip-mean normalisation. Input [B, T, M, 1].
    def build(self, input_shape):
        self.a = self.add_weight(name="a", shape=(int(input_shape[2]),), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        alpha = tf.reshape(tf.sigmoid(self.a), (1, 1, -1, 1))
        return x - alpha * tf.reduce_mean(x, axis=1, keepdims=True)


@register(package="asd")
class L2Norm(L.Layer):
    def call(self, x):
        return tf.math.l2_normalize(x, axis=-1)


@register(package="asd")
class ArcCos(L.Layer):
    # cosine similarity between L2-normalised embeddings and learnable class weights
    def __init__(self, n_classes, **kw):
        super().__init__(**kw)
        self.n_classes = int(n_classes)

    def build(self, input_shape):
        self.W = self.add_weight(name="W", shape=(int(input_shape[-1]), self.n_classes),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):
        return tf.matmul(x, tf.math.l2_normalize(self.W, axis=0))

    def get_config(self):
        return {**super().get_config(), "n_classes": self.n_classes}


@register(package="asd")
class GradReverse(L.Layer):
    # identity in the forward pass, gradient multiplied by -lambda in the backward pass
    def __init__(self, lam=0.0, **kw):
        super().__init__(**kw)
        self._lam0 = float(lam)

    def build(self, input_shape):
        self._lam = self.add_weight(name="lam", shape=(), initializer=tf.keras.initializers.Constant(self._lam0),
                                    trainable=False)
        super().build(input_shape)

    def set_lam(self, v):
        self._lam.assign(float(v))

    def call(self, x):
        lam = tf.cast(self._lam, x.dtype)
        return tf.stop_gradient(x) * (1.0 + lam) - lam * x      # value = x, d/dx = -lam

    def get_config(self):
        return {**super().get_config(), "lam": self._lam0}


def arc_loss_fn(s, m):
    # ArcFace / additive angular margin softmax loss; y_true: class ids, y_pred: cosine similarities
    th, mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def loss(y_true, cos):
        y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        oh = tf.one_hot(y, tf.shape(cos)[-1])
        cos = tf.clip_by_value(cos, -1.0 + 1e-6, 1.0 - 1e-6)
        phi = tf.where(cos > th, tf.cos(tf.acos(cos) + m), cos - mm)
        logits = s * tf.where(oh > 0.5, phi, cos)
        return tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
    return loss


# ------------------------- Baseline 1: autoencoder -------------------------
def build_ae(in_dim, latent):
    def block(x, units):
        x = L.Dense(units, use_bias=False)(x)
        x = L.BatchNormalization()(x)
        return L.ReLU()(x)
    inp = tf.keras.Input(shape=(in_dim,))
    x = inp
    for _ in range(4):
        x = block(x, 128)
    x = block(x, latent)
    for _ in range(4):
        x = block(x, 128)
    return tf.keras.Model(inp, L.Dense(in_dim)(x), name="autoencoder")


def train_ae(Xtr, Xva, cfg, epochs=None):
    model = build_ae(Xtr.shape[1], cfg["AE_LATENT"])
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["AE_LR"]), loss="mse")
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["AE_PATIENCE"], restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
    h = model.fit(Xtr, Xtr, validation_data=(Xva, Xva), epochs=epochs or cfg["AE_EPOCHS"],
                  batch_size=cfg["AE_BATCH"], shuffle=True, callbacks=cbs, verbose=0)
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    print(f"   AE   trained {len(hist['loss'])} epochs | best val_loss {min(hist['val_loss']):.4f}")
    return model, hist


def ae_clip_scores(model, F, cfg, chunk=32):
    out = []
    for i in range(0, len(F), chunk):
        W = make_windows(F[i:i + chunk], cfg["AE_FRAMES"], 1)
        b, w, d = W.shape
        W = W.reshape(-1, d)
        R = model.predict(W, batch_size=4096, verbose=0)
        out.append(((W - R) ** 2).mean(axis=1).reshape(b, w).mean(axis=1))
    return np.concatenate(out)


def ae_residual_map(model, F1, cfg):
    # time-frequency map of squared reconstruction error for ONE clip F1 [T, M] (overlap-add of window residuals)
    fr = cfg["AE_FRAMES"]
    W = make_windows(F1[None], fr, 1)[0]
    R = model.predict(W, batch_size=4096, verbose=0)
    res = ((W - R) ** 2).reshape(len(W), fr, -1)
    T, M = F1.shape
    acc, cnt = np.zeros((T, M)), np.zeros((T, 1))
    for i in range(len(W)):
        acc[i:i + fr] += res[i]
        cnt[i:i + fr] += 1
    return acc / np.maximum(cnt, 1)


# ------------------------- crop-based model trainer -------------------------
def fit_crop_model(model, Ftr, Ytr, Fva, Yva, cfg, loss, aug=False, epochs=None, seed=0, repeat=None,
                   loss_weights=None, metrics=None, grl=None, lam_max=0.0):
    # Ytr/Yva: int array (single output) or dict output_name -> int array. repeat: per-clip repetition (oversampling).
    rng = np.random.default_rng(cfg["SEED"] + seed)
    E, C, k, B = epochs or cfg["CLF_EPOCHS"], cfg["CLF_CROP"], cfg["CLF_CROPS_PER_CLIP"], cfg["CLF_BATCH"]
    multi = isinstance(Ytr, dict)
    base = np.repeat(np.arange(len(Ftr)), repeat) if repeat is not None else np.arange(len(Ftr))
    Xv, kv = fixed_crops(Fva, C, cfg["CLF_INFER_STRIDE"])
    Xv = Xv.reshape(-1, C, Xv.shape[-1])[..., None]
    Yv = {n: np.repeat(a, kv) for n, a in Yva.items()} if multi else np.repeat(Yva, kv)
    total = int(math.ceil(len(base) * k / B)) * E
    lr = tf.keras.optimizers.schedules.CosineDecay(cfg["CLF_LR"], total, alpha=0.01)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, loss_weights=loss_weights, metrics=metrics)
    hist, best, best_w = {}, np.inf, None
    for ep in range(E):
        if grl is not None:                                   # DANN schedule: lambda ramps 0 -> lam_max
            p = ep / max(E - 1, 1)
            grl.set_lam(lam_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))
        Xe, _ = random_crops(Ftr, np.zeros(len(Ftr), int), k, C, rng, idx=base)
        if ep == 0:
            describe("model input BEFORE augmentation", Xe)
        if aug:
            Xe = augment_crops(Xe, rng)
        Ye = {n: np.repeat(a[base], k) for n, a in Ytr.items()} if multi else np.repeat(Ytr[base], k)
        if ep == 0:
            describe("model input AFTER augmentation (fed to net)" if aug else "model input (fed to net, no aug)", Xe)
            for n_, y_ in (Ye.items() if multi else [("labels", Ye)]):
                u_, c_ = np.unique(y_, return_counts=True)
                dbg(f"   training {n_}: {len(u_)} classes | samples per class min/median/max = {c_.min()}/{int(np.median(c_))}/{c_.max()}")
            dbg(f"   validation crops {Xv.shape} | repeat/oversampling used: {repeat is not None}")
        h = model.fit(Xe[..., None], Ye, batch_size=B, epochs=1, shuffle=True, validation_data=(Xv, Yv), verbose=0)
        for key, vals in h.history.items():
            hist.setdefault(key, []).append(float(vals[0]))
        sel = hist["val_arc_loss"][-1] if "val_arc_loss" in hist else hist["val_loss"][-1]
        if sel < best:
            best, best_w = sel, model.get_weights()
    model.set_weights(best_w)
    first, last = (hist.get("val_arc_loss") or hist["val_loss"])[0], (hist.get("val_arc_loss") or hist["val_loss"])[-1]
    dbg(f"   training finished: held-out loss first epoch {first:.4f} -> last {last:.4f} | best {best:.4f}")
    return hist, best


# ------------------------- Baseline 2: MobileNetV2 section classifier -------------------------
def build_mnv2(crop, n_mels, n_sections):
    inp = tf.keras.Input(shape=(crop, n_mels, 1))
    x = L.Concatenate()([inp, inp, inp])
    base = tf.keras.applications.MobileNetV2(input_shape=(crop, n_mels, 3), alpha=0.5, include_top=False,
                                             weights=None, pooling="avg")
    out = L.Dense(n_sections, activation="softmax")(base(x))
    return tf.keras.Model(inp, out, name="mobilenetv2_baseline")


def train_mnv2(Ftr, ytr, Fva, yva, n_sections, cfg, epochs=None, seed=0):
    model = build_mnv2(cfg["CLF_CROP"], Ftr.shape[2], n_sections)
    hist, best = fit_crop_model(model, Ftr, ytr, Fva, yva, cfg, loss="sparse_categorical_crossentropy", aug=False,
                                epochs=epochs, seed=seed, metrics=["accuracy"])
    print(f"   MNv2 trained | best val_loss {best:.4f}")
    if best > 0.95 * math.log(n_sections):
        dbg(f"   WARNING: MobileNetV2 held-out loss {best:.3f} is at chance level (ln {n_sections} = {math.log(n_sections):.3f}) "
            f"although the training loss is {hist['loss'][-1]:.3f} -> the net does not generalise / behaves differently at inference; "
            f"its scores will be (almost) constant and AUC about 0.5")
    return model, hist


def clip_probs(model, F, cfg, chunk=128):
    C, outs = cfg["CLF_CROP"], []
    for i in range(0, len(F), chunk):
        Xc, k = fixed_crops(F[i:i + chunk], C, cfg["CLF_INFER_STRIDE"])
        b = Xc.shape[0]
        outs.append(model.predict(Xc.reshape(b * k, C, -1)[..., None], batch_size=512, verbose=0).reshape(b, k, -1))
    return np.concatenate(outs)


# ------------------------- Baseline 3 and proposed: (DA-)ArcNet -------------------------
def lite_backbone(x):
    x = L.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    x = L.MaxPooling2D(2)(x)
    for i, w in enumerate((32, 64, 128)):
        for j in range(2):
            x = L.SeparableConv2D(w, 3, padding="same", use_bias=False)(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU(name="last_conv")(x) if (i == 2 and j == 1) else L.ReLU()(x)
        if i < 2:
            x = L.MaxPooling2D(2)(x)
    return x


def build_arc_net(crop, n_mels, n_classes, use_ada, use_dann, cfg):
    inp = tf.keras.Input(shape=(crop, n_mels, 1), name="x")
    x = AdaCMN(name="adacmn")(inp) if use_ada else inp
    x = lite_backbone(x)
    e = L.Dense(cfg["EMB_DIM"], use_bias=False)(L.GlobalAveragePooling2D()(x))
    emb = L2Norm(name="emb")(L.BatchNormalization()(e))
    emb_model = tf.keras.Model(inp, emb, name="embedding_model")
    arc = ArcCos(n_classes, name="arc")(emb)
    grl = None
    if use_dann:
        grl = GradReverse(0.0, name="grl")
        d = L.Dense(32, activation="relu")(grl(emb))
        dom = L.Dense(2, activation="softmax", name="dom")(d)
        train_model = tf.keras.Model(inp, [arc, dom], name="da_arcnet")
    else:
        train_model = tf.keras.Model(inp, arc, name="arcnet")
    return train_model, emb_model, grl


def train_arc(Ftr, ytr, dtr, Fva, yva, dva, n_classes, cfg, use_ada, use_dann, epochs=None, seed=0):
    # returns (embedding model, history). Target-domain clips are oversampled in every ArcFace variant.
    train_model, emb_model, grl = build_arc_net(cfg["CLF_CROP"], Ftr.shape[2], n_classes, use_ada, use_dann, cfg)
    repeat = np.where(dtr == 1, cfg["TARGET_OVERSAMPLE"], 1)
    arc_l = arc_loss_fn(cfg["ARC_S"], cfg["ARC_M"])
    if use_dann:
        Ytr, Yva = {"arc": ytr, "dom": dtr}, {"arc": yva, "dom": dva}
        loss, lw = {"arc": arc_l, "dom": "sparse_categorical_crossentropy"}, {"arc": 1.0, "dom": 1.0}
    else:
        Ytr, Yva, loss, lw = ytr, yva, arc_l, None
    hist, best = fit_crop_model(train_model, Ftr, Ytr, Fva, Yva, cfg, loss=loss, aug=cfg["AUG"], epochs=epochs,
                                seed=seed, repeat=repeat, loss_weights=lw, grl=grl, lam_max=cfg["DANN_LAMBDA"])
    print(f"   ARC  trained (ada={use_ada}, dann={use_dann}) | best val arc-loss {best:.4f}")
    return emb_model, hist


def embed_clips(model, F, cfg, chunk=128):
    # L2-normalised crop embeddings [N, crops, D]
    C, outs = cfg["CLF_CROP"], []
    for i in range(0, len(F), chunk):
        Xc, k = fixed_crops(F[i:i + chunk], C, cfg["CLF_INFER_STRIDE"])
        b = Xc.shape[0]
        outs.append(model.predict(Xc.reshape(b * k, C, -1)[..., None], batch_size=512, verbose=0).reshape(b, k, -1))
    return np.concatenate(outs).astype(np.float32)


def save_keras(model, base):
    base = str(base)
    for label, fn in [(".keras", lambda: model.save(base + ".keras")),
                      (".weights.h5", lambda: model.save_weights(base + ".weights.h5")),
                      ("_savedmodel/", lambda: model.export(base + "_savedmodel"))]:
        try:
            with contextlib.redirect_stdout(io.StringIO()):          # Keras prints a very long endpoint list
                fn()
        except Exception as e:
            print(f"   (could not save {label}: {str(e)[:100]})")
dbg_end("models", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ### 6.1 Architecture diagram

# %%
_dbg_tok = dbg_start("arch_diagram")          # --- debug (auto): banner + timer ---
import matplotlib.patches as mp


def draw_architecture(path=None):
    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")
    NEW, OLD, OUT = "#fde7c8", "#e3ecfa", "#dff3e3"

    def box(x, y, w, h, text, fc=OLD, ec="#4a6fa5", fs=8.3):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arr(x1, y1, x2, y2, color="#333", ls="-", rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, ls=ls, connectionstyle=f"arc3,rad={rad}"))

    ax.text(0.5, 56, "DA-ArcNet: training path (top) and inference path (bottom)", fontsize=12, weight="bold")
    # ---- training path ----
    box(0.5, 38, 10, 10, "Raw audio\n10 s, 16 kHz\n(normal clips\nonly)")
    box(13, 38, 12, 10, "Log-mel\n64 mel x T\nglobal z-norm\n(train stats)")
    box(28, 38, 11.5, 10, "Random crop\n64 x 64\n+ SpecAugment\n(target oversampled)")
    box(42.5, 38, 12.5, 10, "AdaCMN  (new)\nlearnable per-mel\nclip-mean blend", fc=NEW, ec="#c77d12")
    box(58, 38, 14, 10, "LiteCNN backbone\nConv + 3 x separable\nblocks, GAP")
    box(75, 38, 10, 10, "Dense(128)\n+ BN + L2\n= embedding e")
    box(88, 46, 11.5, 8.5, "ArcFace head\ncos(theta + m)\nsection x condition", fc=OLD)
    box(88, 31, 11.5, 11, "Gradient reversal\n(-lambda) + domain\nclassifier src/tgt\n(new)", fc=NEW, ec="#c77d12")
    for x1, x2 in [(10.5, 13), (25, 28), (39.5, 42.5), (55, 58), (72, 75)]:
        arr(x1, 43, x2, 43)
    arr(85, 45, 88, 49.5); arr(85, 41, 88, 37)
    # ---- inference path ----
    ax.plot([0.5, 99.5], [30, 30], ls=":", color="#999", lw=1)
    box(21, 17, 19, 10, "Autoencoder branch\n5 frames x 64 mel\n-> s_ae (recon. error)")
    box(52, 17, 24, 10, "Cosine kNN (k=3) vs. normal\nreference bank of the same\nsection  -> s_prop", fc=NEW, ec="#c77d12")
    box(33, 3, 18, 9, "z-score fusion on\nheld-out normals\n-> s_full", fc=NEW, ec="#c77d12")
    box(58, 3, 24, 9, "Gamma 90th-percentile threshold\n(held-out normal clips)\n-> Normal / Anomaly", fc=OUT, ec="#3f8f52")
    arr(19, 38, 28, 27)           # log-mel -> AE branch
    arr(80, 38, 66, 27)           # embedding -> kNN
    arr(30, 17, 40, 12)           # AE -> fusion
    arr(62, 17, 48, 12)           # kNN -> fusion
    arr(51, 7.5, 58, 7.5)         # fusion -> decision
    ax.text(0.5, 33, "TRAINING", fontsize=9, color="#555", weight="bold")
    ax.text(0.5, 27, "INFERENCE", fontsize=9, color="#555", weight="bold")
    ax.text(99.5, 1, "orange = components proposed in this work", fontsize=8, ha="right", color="#c77d12")
    if path is not None:
        plt.savefig(path)
    plt.show()


draw_architecture(FIG_DIR / "architecture.png")
dbg_end("arch_diagram", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 7. Metrics and statistics
# Following the DCASE 2022 Task 2 protocol:
# 
# * **AUC (source)** and **AUC (target)** - computed separately on source- and target-domain test clips of each section.
# * **pAUC** - partial (McClish-standardised) AUC for FPR in [0, 0.1], both domains mixed.
# * **Official score** - harmonic mean of all AUC_source, AUC_target and pAUC values of a machine (all sections).
# * **EER**, **Average Precision**, and **threshold metrics** (Accuracy, Precision, Recall, Specificity, F1, Balanced accuracy, MCC, confusion matrix). The threshold is chosen **without any anomalous or test data**: a gamma distribution is fitted to the scores of *held-out normal source clips* (10 % of the training clips, never used for training or as kNN references) of that section; threshold = its 90th percentile. Target clips are *not* used for it, so target numbers show the real cost of domain shift.
# * **Statistical rigour.** 95 % **bootstrap confidence intervals** (test clips resampled within section / domain / label strata), **multi-seed** repeats (mean +/- std) and **Wilcoxon signed-rank tests** between the proposed model and each baseline over all machine-sections.
# * **Fusion** (`full`): per section, the log-scores of `prop` and `ae` are z-scored with the mean/std of the held-out normal clips, averaged and exponentiated (monotone, always positive).

# %%
_dbg_tok = dbg_start("metrics")          # --- debug (auto): banner + timer ---
from scipy.stats import wilcoxon


def safe_auc(y, s, max_fpr=None):
    y, s = np.asarray(y), np.asarray(s, float)
    if len(np.unique(y)) < 2 or np.isnan(s).any():
        return np.nan
    return roc_auc_score(y, s, max_fpr=max_fpr)


def official_hmean(values):
    v = np.asarray(values, float).ravel()
    v = v[~np.isnan(v)]
    return float(hmean(v)) if len(v) else np.nan


def section_table(df, extra=(), models=None):
    # AUC_source, AUC_target, pAUC per (machine, section, model[, extra]); df = TEST rows only.
    models = models or MODELS
    keys = ["machine", "section"] + list(extra)
    rows = []
    for k, g in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        s, t = g[g.domain == "source"], g[g.domain == "target"]
        for m in models:
            rows.append(dict(zip(keys, k), model=m, AUC_source=safe_auc(s.label, s[m]),
                             AUC_target=safe_auc(t.label, t[m]), pAUC=safe_auc(g.label, g[m], CFG["MAX_FPR"])))
    return pd.DataFrame(rows)


def machine_table(sec_df, extra=()):
    keys = ["machine", "model"] + list(extra)
    rows = []
    for k, g in sec_df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        rows.append(dict(zip(keys, k), AUC_source=g.AUC_source.mean(), AUC_target=g.AUC_target.mean(), pAUC=g.pAUC.mean(),
                         official=official_hmean(g[["AUC_source", "AUC_target", "pAUC"]].values)))
    return pd.DataFrame(rows)


def overall_table(sec_df):
    rows = []
    for m, g in sec_df.groupby("model"):
        rows.append(dict(model=m, AUC_source=g.AUC_source.mean(), AUC_target=g.AUC_target.mean(), pAUC=g.pAUC.mean(),
                         official=official_hmean(g[["AUC_source", "AUC_target", "pAUC"]].values)))
    return pd.DataFrame(rows)


def curve_table(df, models=None):
    rows = []
    for mc, g in df.groupby("machine"):
        for m in (models or MODELS):
            fpr, tpr, _ = roc_curve(g.label, g[m])
            i = np.nanargmin(np.abs(fpr - (1 - tpr)))
            rows.append(dict(machine=mc, model=m, EER=(fpr[i] + 1 - tpr[i]) / 2, AP=average_precision_score(g.label, g[m])))
    return pd.DataFrame(rows)


# ---------------- fusion (prop + ae) ----------------
def compute_refs(df):
    # per section: mean/std of log-scores of held-out normal clips, for 'prop' and 'ae'
    refs = {}
    for s, g in df[df.split == "val"].groupby("section"):
        lp, la = np.log(g["prop"].values + 1e-8), np.log(g["ae"].values + 1e-8)
        refs[int(s)] = (float(lp.mean()), float(max(lp.std(), 0.05)), float(la.mean()), float(max(la.std(), 0.05)))
    return refs


def fuse(prop, ae, ref):
    mu_p, sd_p, mu_a, sd_a = ref
    zp = (np.log(np.asarray(prop, float) + 1e-8) - mu_p) / sd_p
    za = (np.log(np.asarray(ae, float) + 1e-8) - mu_a) / sd_a
    return np.exp(np.clip((zp + za) / 2.0, -100, 600))       # wide clip = overflow guard only (no ties under noise)


def add_fusion(df, refs):
    df = df.copy()
    df["full"] = np.nan
    for s, ref in refs.items():
        m = (df["section"] == s).values
        df.loc[m, "full"] = fuse(df.loc[m, "prop"].values, df.loc[m, "ae"].values, ref)
    return df


# ---------------- thresholds and threshold-based metrics ----------------
def fit_thresholds(df, models=None):
    thr = {}
    for sec, g in df[df.split == "val"].groupby("section"):
        for m in (models or MODELS):
            x = g[m].values.astype(float)
            x = x[np.isfinite(x)]
            if len(x) < 3:
                continue
            try:
                a, loc, sc = gamma.fit(x, floc=0)
                thr[(int(sec), m)] = float(gamma.ppf(CFG["THR_PERCENTILE"], a, loc, sc))
            except Exception:
                thr[(int(sec), m)] = float(np.quantile(x, CFG["THR_PERCENTILE"]))
    return thr


def add_predictions(df, models=None):
    models = models or MODELS
    outs = []
    for mc, g in df.groupby("machine"):
        g = g.copy()
        thr = fit_thresholds(g, models)
        for m in models:
            g[f"pred_{m}"] = g[m].values > g["section"].map({s: thr[(s, m)] for s in g.section.unique() if (s, m) in thr}).values
        outs.append(g)
    return pd.concat(outs, ignore_index=True)


def cls_metrics(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, p, labels=[0, 1]).ravel()]
    n = tn + fp + fn + tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mcc = float(matthews_corrcoef(y, p)) if (len(np.unique(y)) > 1 and len(np.unique(p)) > 1) else 0.0
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, Accuracy=(tp + tn) / n, Precision=prec, Recall=rec, Specificity=spec,
                F1=f1, BalancedAcc=(rec + spec) / 2, MCC=mcc, FPR=1 - spec, FNR=1 - rec)


def classification_table(df_test, models=None):
    rows = []
    scopes = {"all": lambda g: g, "source": lambda g: g[g.domain == "source"], "target": lambda g: g[g.domain == "target"]}
    for mc, g in list(df_test.groupby("machine")) + [("ALL", df_test)]:
        for m in (models or MODELS):
            for sc, fn in scopes.items():
                gg = fn(g)
                rows.append(dict(machine=mc, model=m, domain=sc, **cls_metrics(gg.label.values, gg[f"pred_{m}"].values)))
    return pd.DataFrame(rows)


# ---------------- bootstrap confidence intervals ----------------
def bootstrap_machine(g, models, n_boot, seed=0):
    # resample test clips within (section, domain, label) strata; returns {model: [n_boot, n_sections, 3]}
    rng = np.random.default_rng(seed)
    g = g.reset_index(drop=True)
    secs = sorted(g.section.unique())
    strata = [np.where((g.section == s).values & (g.domain == d).values & (g.label == l).values)[0]
              for s in secs for d in ("source", "target") for l in (0, 1)]
    strata = [ix for ix in strata if len(ix)]
    Y, DOM, SEC = g.label.values, g.domain.values, g.section.values
    S = {m: g[m].values.astype(float) for m in models}
    res = {m: np.full((n_boot, len(secs), 3), np.nan) for m in models}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])
        yb, db, sb = Y[idx], DOM[idx], SEC[idx]
        for m in models:
            sm = S[m][idx]
            for j, s in enumerate(secs):
                ms = sb == s
                for t, dom in enumerate(("source", "target")):
                    mm = ms & (db == dom)
                    res[m][b, j, t] = safe_auc(yb[mm], sm[mm])
                res[m][b, j, 2] = safe_auc(yb[ms], sm[ms], CFG["MAX_FPR"])
    return res


def ci_table(test_df, models=None, n_boot=None, seed=0):
    # 95 % bootstrap CI of official / AUC_source / AUC_target / pAUC per machine and overall
    models, n_boot = models or MODELS, n_boot or CFG["BOOT_N"]
    by_machine = {mc: bootstrap_machine(g, models, n_boot, seed) for mc, g in test_df.groupby("machine")}

    def summ(a):
        return dict(official=np.array([official_hmean(a[b]) for b in range(a.shape[0])]),
                    AUC_source=np.nanmean(a[:, :, 0], axis=1), AUC_target=np.nanmean(a[:, :, 1], axis=1),
                    pAUC=np.nanmean(a[:, :, 2], axis=1))

    rows = []
    for m in models:
        groups = {mc: r[m] for mc, r in by_machine.items()}
        groups["ALL"] = np.concatenate(list(groups.values()), axis=1)
        for mc, a in groups.items():
            for metric, arr in summ(a).items():
                rows.append(dict(machine=mc, model=m, metric=metric, mean=float(np.nanmean(arr)),
                                 lo=float(np.nanpercentile(arr, 2.5)), hi=float(np.nanpercentile(arr, 97.5))))
    return pd.DataFrame(rows)


def wilcoxon_table(sec_tab, candidates=("prop", "full"), baselines=("ae", "mnv2", "arc")):
    # paired Wilcoxon signed-rank test over (machine, section) pairs: candidate > baseline ?
    t = sec_tab.copy()
    t["official"] = [official_hmean(r) for r in t[["AUC_source", "AUC_target", "pAUC"]].values]
    rows = []
    for c in candidates:
        for b in baselines:
            for metric in ("official", "AUC_source", "AUC_target", "pAUC"):
                pc = t[t.model == c].set_index(["machine", "section"])[metric]
                pb = t[t.model == b].set_index(["machine", "section"])[metric]
                d = (pc - pb).dropna()
                try:
                    p_g, p_2 = wilcoxon(d, alternative="greater")[1], wilcoxon(d)[1]
                except Exception:
                    p_g = p_2 = np.nan
                rows.append(dict(candidate=c, baseline=b, metric=metric, n_pairs=len(d),
                                 mean_diff=float(d.mean()) if len(d) else np.nan, p_greater=p_g, p_two_sided=p_2))
    return pd.DataFrame(rows)


# ---------------- LaTeX helper for the paper ----------------
def _tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def latex_table(df, caption, label, fmt="{:.3f}", bold="row_max", index_name="", col_spec=None):
    # df: index = row labels, columns = numeric (or strings). bold = 'row_max' | 'col_max' | 'row_min' | 'col_min' | None
    d = df.copy()
    num = d.apply(pd.to_numeric, errors="coerce")
    best = None
    if bold in ("row_max", "row_min"):
        best = num.eq(num.max(axis=1) if bold == "row_max" else num.min(axis=1), axis=0)
    elif bold in ("col_max", "col_min"):
        best = num.eq(num.max(axis=0) if bold == "col_max" else num.min(axis=0), axis=1)
    cols = list(d.columns)
    spec = col_spec or ("l" + "c" * len(cols))
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption, r"\label{%s}" % label, r"\small",
             r"\adjustbox{max width=\columnwidth}{%", r"\begin{tabular}{%s}" % spec, r"\toprule",
             " & ".join([_tex(index_name)] + [_tex(c) for c in cols]) + r" \\", r"\midrule"]
    for r in d.index:
        cells = []
        for c in cols:
            v = d.loc[r, c]
            txt = fmt.format(v) if isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v) else ("--" if pd.isna(v) else _tex(v))
            if best is not None and bool(best.loc[r, c]) and not pd.isna(num.loc[r, c]):
                txt = r"\textbf{%s}" % txt
            cells.append(txt)
        lines.append(" & ".join([_tex(r)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"
dbg_end("metrics", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 8. Full pipeline for one machine
# `process_machine()`: load features -> split (held-out normal source clips; **no test clip is ever used for training, normalisation, thresholds or kNN references**) -> fit normalisation on training clips -> train all models -> score held-out + test clips -> robustness to additive noise -> save models, reference banks, embeddings, scores, training curves, complexity numbers.

# %%
_dbg_tok = dbg_start("pipeline")          # --- debug (auto): banner + timer ---
def process_machine(machine, seed=None, save=True, noise=True, norm_mode=None, epoch_scale=1.0, ablation=None):
    # save: True (models + tables) | 'scores' (scores csv only, used for extra seeds) | False
    cfg = CFG
    seed = cfg["SEED"] if seed is None else seed
    norm_mode = norm_mode or cfg["NORM_MODE"]
    ablation = (machine in cfg["ABLATION_MACHINES"]) if ablation is None else ablation
    ae_ep = max(1, int(round(cfg["AE_EPOCHS"] * epoch_scale)))
    clf_ep = max(1, int(round(cfg["CLF_EPOCHS"] * epoch_scale)))
    set_seed(seed)
    t0 = time.time()

    meta, X = get_data(machine)
    tr, va, te = split_indices(meta, cfg["SEED"])              # identical split for every seed
    print(f"[{machine}] seed {seed} | train {len(tr)} | held-out normal {len(va)} | test {len(te)} | norm={norm_mode}")
    dbg(f"[{machine}] split sizes: train {len(tr)} (source {int(((meta.domain.values[tr]) == 'source').sum())}, "
        f"target {int(((meta.domain.values[tr]) == 'target').sum())}) | held-out {len(va)} | test {len(te)} "
        f"(normal {int((meta.label.values[te] == 0).sum())}, anomaly {int((meta.label.values[te] == 1).sum())})")
    st = fit_norm(X[tr], norm_mode)
    F = apply_norm(X, st)
    T = X.shape[1]
    dbg(f"[{machine}] BEFORE vs AFTER pre-processing (training clips):")
    describe("   BEFORE: raw log-mel [dB]", X[tr])
    describe(f"   AFTER : normalised ('{norm_mode}')", F[tr])
    dbg(f"   normalisation statistics come from TRAINING clips only: mu[:3]={np.round(st['mu'][:3], 2)} sd[:3]={np.round(st['sd'][:3], 2)}")

    # ---- labels
    classes = sorted(meta[meta.split == "train"].cls_key.unique())
    cidx = {k: i for i, k in enumerate(classes)}
    y_all = meta.cls_key.map(cidx).values
    ytr, yva = y_all[tr].astype(int), y_all[va].astype(int)
    dbg(f"[{machine}] classes (section|attributes): {len(classes)} -> {classes[:6]}{' ...' if len(classes) > 6 else ''}")
    dom = (meta.domain == "target").astype(int).values
    secs = meta.section.values
    n_sec = int(secs.max()) + 1

    # ---- Baseline 1: autoencoder
    D = cfg["AE_FRAMES"] * cfg["N_MELS"]
    Wtr = make_windows(F[tr], cfg["AE_FRAMES"], cfg["AE_TRAIN_STRIDE"]).reshape(-1, D)
    Wva = make_windows(F[va], cfg["AE_FRAMES"], cfg["AE_TRAIN_STRIDE"]).reshape(-1, D)
    describe("autoencoder input windows (train)", Wtr)
    ae, h_ae = train_ae(Wtr, Wva, cfg, epochs=ae_ep)
    del Wtr, Wva

    # ---- Baseline 2: MobileNetV2 section classifier
    mnv2, h_mn = train_mnv2(F[tr], secs[tr], F[va], secs[va], n_sec, cfg, epochs=clf_ep, seed=seed)

    # ---- Baseline 3, proposed model and ablation variants
    variants = [("arc", False, False), ("prop", True, True)]
    if ablation:
        variants += [("arc_ada", True, False), ("arc_dann", False, True)]
    emb, hist = {}, {"ae": h_ae, "mnv2": h_mn}
    for name, use_ada, use_dann in variants:
        emb[name], hist[name] = train_arc(F[tr], ytr, dom[tr], F[va], yva, dom[va], len(classes), cfg,
                                          use_ada, use_dann, epochs=clf_ep, seed=seed)
    sec_tr = secs[tr]
    refs_emb = {n: embed_clips(m, F[tr], cfg) for n, m in emb.items()}      # normal reference banks (training clips only)

    def score_clips(Fs, sub):
        cols = {"ae": ae_clip_scores(ae, Fs, cfg),
                "mnv2": section_scores_from_probs(clip_probs(mnv2, Fs, cfg), sub.section.values)}
        for n, m in emb.items():
            cols[n] = knn_scores(embed_clips(m, Fs, cfg), sub.section.values, refs_emb[n], sec_tr, cfg["KNN_K"])
        return cols

    def make_rows(idx, split_name, Fs):
        sub = meta.iloc[idx]
        d = sub[["name", "section", "domain", "label"]].copy().reset_index(drop=True)
        d["split"], d["machine"] = split_name, machine
        for n, v in score_clips(Fs, sub).items():
            d[n] = v
        return d

    scores = pd.concat([make_rows(va, "val", F[va]), make_rows(te, "test", F[te])], ignore_index=True)
    dbg(f"[{machine}] anomaly-score sanity check (a constant score column means the model failed -> AUC 0.5):")
    for m_ in [c for c in ("ae", "mnv2", "arc", "prop", "arc_ada", "arc_dann") if c in scores.columns]:
        v_ = scores.loc[scores.split == "test", m_].values
        st_ = describe(f"   score[{m_}] (test)", v_, warn=False)
        if st_["nan"] or st_["inf"]:
            dbg(f"   WARNING: score[{m_}] has NaN/Inf")
        elif st_["std"] < 1e-9 * max(1.0, abs(st_["mean"])):
            dbg(f"   WARNING: score[{m_}] is CONSTANT on the test set -> AUC will be exactly 0.5 for this model")
    refs = compute_refs(scores)
    scores = add_fusion(scores, refs)
    describe("score[full] (fusion, test)", scores.loc[scores.split == "test", "full"].values)

    # ---- robustness to additive noise (test clips only; models/statistics fixed)
    noise_df = None
    if noise and cfg["NOISE_SNRS"]:
        waves = list(tqdm(map(read_wave, meta.iloc[te].path), total=len(te), desc=f"noise study [{machine}]"))
        rows = []
        for kind in cfg["NOISE_KINDS"]:
            for snr in cfg["NOISE_SNRS"]:
                rn = np.random.default_rng(cfg["SEED"] + int(snr))
                Xn = np.stack([logmel(add_noise(w, snr, kind, rn))[:T] for w in waves])
                d = make_rows(te, "test", apply_norm(Xn, st))
                d["snr"], d["noise"] = snr, kind
                rows.append(d)
        noise_df = add_fusion(pd.concat(rows, ignore_index=True), refs)
        del waves

    # ---- complexity: parameters and latency per 10-s clip (after one warm-up call)
    n_t = min(20, len(te))
    Fs, sub = F[te[:n_t]], meta.iloc[te[:n_t]]
    lat = {}

    def timeit(fn):
        fn()
        t = time.time()
        fn()
        return (time.time() - t) / n_t
    lat["ae"] = timeit(lambda: ae_clip_scores(ae, Fs, cfg))
    lat["mnv2"] = timeit(lambda: clip_probs(mnv2, Fs, cfg))
    for n, m in emb.items():
        lat[n] = timeit(lambda m=m, n=n: knn_scores(embed_clips(m, Fs, cfg), sub.section.values, refs_emb[n], sec_tr, cfg["KNN_K"]))
    params = {"ae": int(ae.count_params()), "mnv2": int(mnv2.count_params()), **{n: int(m.count_params()) for n, m in emb.items()}}
    complexity = dict(params=params, sec_per_clip=lat)

    res = dict(machine=machine, seed=seed, scores=scores, noise=noise_df, hist=hist,
               thresholds={f"{s}|{m}": t for (s, m), t in fit_thresholds(scores, [x for x in ALL_MODELS if x in scores.columns and scores[x].notna().any()]).items()},
               refs={str(s): list(v) for s, v in refs.items()}, classes=classes, complexity=complexity,
               seconds=time.time() - t0)

    if save:
        tag = "" if seed == cfg["SEED"] else f"_seed{seed}"
        scores.to_csv(TAB_DIR / f"scores_{machine}{tag}.csv", index=False)
    if save is True:
        if noise_df is not None:
            noise_df.to_csv(TAB_DIR / f"noise_scores_{machine}.csv", index=False)
        with open(TAB_DIR / f"history_{machine}.json", "w") as f:
            json.dump(hist, f)
        with open(TAB_DIR / f"complexity_{machine}.json", "w") as f:
            json.dump(complexity, f)
        np.savez(MODEL_DIR / f"{machine}_norm.npz", mode=st["mode"], mu=st["mu"], sd=st["sd"])
        np.savez_compressed(MODEL_DIR / f"{machine}_refs.npz", sections=sec_tr, **{n: r for n, r in refs_emb.items() if n in ("arc", "prop")})
        # clip-level embeddings of all clips (for the t-SNE figure)
        np.savez_compressed(TAB_DIR / f"emb_{machine}.npz", name=meta.name.values, section=secs, domain=meta.domain.values,
                            split=meta.split.values, label=meta.label.values,
                            **{f"emb_{n}": embed_clips(emb[n], F, cfg).mean(axis=1) for n in ("arc", "prop")})
        with open(MODEL_DIR / f"{machine}_info.json", "w") as f:
            json.dump(dict(machine=machine, classes=classes, thresholds=res["thresholds"], refs=res["refs"], complexity=complexity,
                           config={k: v for k, v in cfg.items() if not isinstance(v, Path)}), f, indent=1)
        save_keras(ae, MODEL_DIR / f"{machine}_autoencoder")
        save_keras(mnv2, MODEL_DIR / f"{machine}_mobilenetv2")
        save_keras(emb["arc"], MODEL_DIR / f"{machine}_arcface_knn")
        save_keras(emb["prop"], MODEL_DIR / f"{machine}_proposed_dannet")
        print(f"   models + tables saved to {OUT_DIR}")
    print(f"[{machine}] done in {res['seconds'] / 60:.1f} min")
    del F, X
    return res
dbg_end("pipeline", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 9. Train and evaluate every machine
# Each machine is downloaded, processed and trained one after another (AE, MobileNetV2, ArcFace-kNN, proposed DA-ArcNet); scores, training curves, embeddings and models are written to Drive as soon as a machine finishes. If the session disconnects, simply run this cell again - finished machines are skipped.

# %%
_dbg_tok = dbg_start("run_main")          # --- debug (auto): banner + timer ---
import gc
need_noise = bool(CFG["NOISE_SNRS"])
for mc in CFG["RUN_MACHINES"]:
    sf, nf, hf = TAB_DIR / f"scores_{mc}.csv", TAB_DIR / f"noise_scores_{mc}.csv", TAB_DIR / f"history_{mc}.json"
    if CFG["RESUME"] and sf.exists() and hf.exists() and (nf.exists() or not need_noise):
        print(f"[{mc}] saved results found -> skipping")
        continue
    dbg(f"run_main: training machine '{mc}' ...")
    res = process_machine(mc)
    dbg(f"run_main: '{mc}' finished -> files now on Drive: {sorted(p.name for p in TAB_DIR.glob(f'*{mc}*'))[:6]}")
    del res
    gc.collect()
    tf.keras.backend.clear_session()
print("All machines finished.")
dbg_end("run_main", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 10. Results tables, confidence intervals and significance tests

# %%
_dbg_tok = dbg_start("tables")          # --- debug (auto): banner + timer ---
def load_all():
    scores = pd.concat([pd.read_csv(TAB_DIR / f"scores_{m}.csv") for m in CFG["RUN_MACHINES"]], ignore_index=True)
    nfiles = [TAB_DIR / f"noise_scores_{m}.csv" for m in CFG["RUN_MACHINES"] if (TAB_DIR / f"noise_scores_{m}.csv").exists()]
    noise = pd.concat([pd.read_csv(f) for f in nfiles], ignore_index=True) if nfiles else None
    hist = {m: json.load(open(TAB_DIR / f"history_{m}.json")) for m in CFG["RUN_MACHINES"]}
    return scores, noise, hist


scores_all, noise_all, hist_all = load_all()
MACHINES = CFG["RUN_MACHINES"]
test_all = scores_all[scores_all.split == "test"].reset_index(drop=True)

sec_tab = section_table(test_all)
mach_tab = machine_table(sec_tab)
over_tab = overall_table(sec_tab)
curve_tab = curve_table(test_all)
pred_all = add_predictions(scores_all)
pred_test = pred_all[pred_all.split == "test"].reset_index(drop=True)
cls_tab = classification_table(pred_test)
dbg(f"scores_all {scores_all.shape} | test rows {len(test_all)} | machines {MACHINES}")
for _m in MODELS:
    _v = test_all[_m].values
    dbg(f"   score column '{_m}': NaN={int(np.isnan(_v).sum())} | unique values={len(np.unique(_v))} | range {np.nanmin(_v):.4g} .. {np.nanmax(_v):.4g}")
    if len(np.unique(_v)) < 0.2 * len(_v):
        dbg(f"   WARNING: '{_m}' has very few distinct scores -> many ties / a collapsed model (check AUC = 0.5000 in the tables)")
_auc05 = sec_tab[(sec_tab.AUC_source == 0.5) | (sec_tab.AUC_target == 0.5)]
if len(_auc05):
    dbg(f"WARNING: {len(_auc05)} machine-section-model entries have AUC exactly 0.5000 (constant scores): models {sorted(_auc05.model.unique())}, "
        f"machines {sorted(_auc05.machine.unique())}")


def wide(tab, metric):
    w = tab.pivot(index="machine", columns="model", values=metric)[MODELS].reindex(MACHINES)
    w.loc["MEAN"] = w.mean()
    return w.rename(columns=MODEL_NAMES).round(4)


print("=== Threshold-free metrics per machine (mean over the 3 sections) ===")
for metric, title in [("AUC_source", "AUC - source domain"), ("AUC_target", "AUC - target domain (domain-shifted)"),
                      ("pAUC", f"pAUC (FPR <= {CFG['MAX_FPR']}), both domains"), ("official", "DCASE official score (harmonic mean)")]:
    print(f"\n{title}")
    display(wide(mach_tab, metric))

print("\n=== Overall (all machines, all sections) ===")
display(over_tab.assign(model=over_tab.model.map(MODEL_NAMES)).round(4))

print("\n=== EER and Average Precision per machine ===")
display(curve_tab.pivot(index="machine", columns="model", values=["EER", "AP"]).reindex(MACHINES).round(4))

print("\n=== Threshold-based metrics (all machines pooled) ===")
display(cls_tab[cls_tab.machine == "ALL"].assign(model=lambda d: d.model.map(MODEL_NAMES))
        .drop(columns="machine").round(4))

print("\n=== Threshold-based metrics per machine (both domains) ===")
display(cls_tab[(cls_tab.domain == "all") & (cls_tab.machine != "ALL")].assign(model=lambda d: d.model.map(MODEL_NAMES))
        .drop(columns="domain").round(4))

sec_tab.assign(model=sec_tab.model.map(MODEL_NAMES)).to_csv(TAB_DIR / "metrics_per_section.csv", index=False)
mach_tab.assign(model=mach_tab.model.map(MODEL_NAMES)).to_csv(TAB_DIR / "metrics_per_machine.csv", index=False)
over_tab.assign(model=over_tab.model.map(MODEL_NAMES)).to_csv(TAB_DIR / "metrics_overall.csv", index=False)
curve_tab.assign(model=curve_tab.model.map(MODEL_NAMES)).to_csv(TAB_DIR / "metrics_eer_ap.csv", index=False)
cls_tab.assign(model=cls_tab.model.map(MODEL_NAMES)).to_csv(TAB_DIR / "metrics_threshold_based.csv", index=False)
print("\nTables saved to", TAB_DIR)
dbg_end("tables", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %%
_dbg_tok = dbg_start("stats_ci")          # --- debug (auto): banner + timer ---
# 95 % bootstrap confidence intervals (test clips resampled within section / domain / label strata)
dbg(f"bootstrap: {CFG['BOOT_N']} resamples x {len(MODELS)} models x {len(MACHINES)} machines")
ci_tab = ci_table(test_all, MODELS, CFG["BOOT_N"], seed=CFG["SEED"])
dbg(f"bootstrap done: {len(ci_tab)} rows")
ci_tab.assign(model=ci_tab.model.map(MODEL_NAMES)).to_csv(TAB_DIR / "bootstrap_ci.csv", index=False)
print(f"=== Bootstrap 95 % CI ({CFG['BOOT_N']} resamples), all machines pooled ===")
t = ci_tab[ci_tab.machine == "ALL"].copy()
t["value"] = t.apply(lambda r: f"{r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1)
display(t.pivot(index="model", columns="metric", values="value").loc[MODELS].rename(index=MODEL_NAMES)
        [["AUC_source", "AUC_target", "pAUC", "official"]])

# paired Wilcoxon signed-rank tests over all (machine, section) pairs
wil_tab = wilcoxon_table(sec_tab)
wil_tab.to_csv(TAB_DIR / "wilcoxon_tests.csv", index=False)
print("\n=== Paired Wilcoxon signed-rank tests: candidate vs baseline over machine-sections (H1: candidate > baseline) ===")
display(wil_tab[wil_tab.metric.isin(["official", "AUC_target"])].round(4))
dbg_end("stats_ci", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 11. Training curves
# Left to right: autoencoder reconstruction loss; MobileNetV2 section-classification loss; ArcFace-kNN loss; proposed model (ArcFace loss and the domain-adversary loss on held-out normal clips).

# %%
_dbg_tok = dbg_start("plot_training")          # --- debug (auto): banner + timer ---
fig, axs = plt.subplots(len(MACHINES), 4, figsize=(19, 2.9 * len(MACHINES)), squeeze=False)
for r, mc in enumerate(MACHINES):
    h = hist_all[mc]
    a = axs[r, 0]; a.plot(h["ae"]["loss"], label="train"); a.plot(h["ae"]["val_loss"], label="held-out normal")
    a.set_title(f"{mc} - AE loss (MSE)"); a.set_xlabel("epoch"); a.legend(fontsize=7)
    a = axs[r, 1]; a.plot(h["mnv2"]["loss"], label="train"); a.plot(h["mnv2"]["val_loss"], label="held-out normal")
    a.set_title(f"{mc} - MobileNetV2 loss"); a.set_xlabel("epoch"); a.legend(fontsize=7)
    a = axs[r, 2]; a.plot(h["arc"]["loss"], label="train"); a.plot(h["arc"]["val_loss"], label="held-out normal")
    a.set_title(f"{mc} - ArcFace-kNN loss"); a.set_xlabel("epoch"); a.legend(fontsize=7)
    a = axs[r, 3]
    hp = h["prop"]
    a.plot(hp["arc_loss"], label="ArcFace train"); a.plot(hp["val_arc_loss"], label="ArcFace held-out")
    a.plot(hp["dom_loss"], "--", label="domain adversary (train)")
    a.set_title(f"{mc} - DA-ArcNet losses"); a.set_xlabel("epoch"); a.legend(fontsize=7)
plt.tight_layout()
savefig("training_curves")
dbg_end("plot_training", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 12. Detection performance per machine and the domain-shift gap

# %%
_dbg_tok = dbg_start("plot_bars")          # --- debug (auto): banner + timer ---
palette = {MODEL_NAMES[m]: MODEL_COLORS[m] for m in MODELS}
plot_df = mach_tab.assign(model=mach_tab.model.map(MODEL_NAMES))
order = MACHINES

fig, axs = plt.subplots(2, 2, figsize=(15, 9))
for ax, (metric, title) in zip(axs.ravel(), [("AUC_source", "AUC - source domain"), ("AUC_target", "AUC - target domain"),
                                             ("pAUC", f"pAUC (FPR <= {CFG['MAX_FPR']})"), ("official", "DCASE official score")]):
    sns.barplot(data=plot_df, x="machine", y=metric, hue="model", order=order, palette=palette, ax=ax)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_ylim(0.4, 1.02); ax.set_title(title); ax.set_xlabel("")
    ax.legend(fontsize=8, loc="lower right") if metric == "AUC_source" else ax.legend_.remove()
plt.tight_layout()
savefig("metrics_per_machine")

# domain-shift gap
gap = mach_tab.assign(gap=mach_tab.AUC_source - mach_tab.AUC_target, model=mach_tab.model.map(MODEL_NAMES))
plt.figure(figsize=(10, 4))
sns.barplot(data=gap, x="machine", y="gap", hue="model", order=order, palette=palette)
plt.axhline(0, color="k", lw=1)
plt.ylabel("AUC(source) - AUC(target)"); plt.xlabel("")
plt.title("Domain-shift penalty: how much detection degrades on the shifted (target) domain")
plt.legend(fontsize=8)
savefig("domain_shift_gap")

# heatmaps
fig, axs = plt.subplots(1, 4, figsize=(24, 4.6))
for ax, metric in zip(axs, ["AUC_source", "AUC_target", "pAUC", "official"]):
    sns.heatmap(mach_tab.pivot(index="machine", columns="model", values=metric)[MODELS].reindex(order).rename(columns=MODEL_NAMES),
                annot=True, fmt=".3f", cmap="viridis", vmin=0.5, vmax=1.0, ax=ax, cbar=False)
    ax.set_title(metric); ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
plt.tight_layout()
savefig("metrics_heatmap")
dbg_end("plot_bars", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 13. ROC and Precision-Recall curves
# Curves pool the three sections of a machine for visualisation (the official numbers above are computed per section). The dashed vertical line marks the pAUC region (FPR <= 0.1).

# %%
_dbg_tok = dbg_start("plot_roc")          # --- debug (auto): banner + timer ---
def grid(n, ncols=4, w=4.3, h=3.7):
    nrows = int(math.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    axs = np.atleast_1d(axs).ravel()
    for a in axs[n:]:
        a.axis("off")
    return fig, axs


# ROC, domains mixed
fig, axs = grid(len(MACHINES))
for ax, mc in zip(axs, MACHINES):
    g = test_all[test_all.machine == mc]
    for m in MODELS:
        fpr, tpr, _ = roc_curve(g.label, g[m])
        ax.plot(fpr, tpr, color=MODEL_COLORS[m], label=f"{MODEL_NAMES[m]} ({roc_auc_score(g.label, g[m]):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.axvline(CFG["MAX_FPR"], color="gray", ls="--", lw=0.8)
    ax.set_title(mc); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=7, loc="lower right")
fig.suptitle("ROC curves (source + target test clips)", y=1.01)
plt.tight_layout()
savefig("roc_curves")

# ROC, source vs target
fig, axs = grid(len(MACHINES))
for i, (ax, mc) in enumerate(zip(axs, MACHINES)):
    g = test_all[test_all.machine == mc]
    for m in MODELS:
        for dom, ls in [("source", "-"), ("target", "--")]:
            gg = g[g.domain == dom]
            fpr, tpr, _ = roc_curve(gg.label, gg[m])
            ax.plot(fpr, tpr, color=MODEL_COLORS[m], ls=ls, lw=1.6,
                    label=f"{MODEL_NAMES[m]} - {dom} ({roc_auc_score(gg.label, gg[m]):.2f})" if i == 0 else None)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_title(mc); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
axs[0].legend(fontsize=6, loc="lower right")
fig.suptitle("ROC: source (solid) vs target (dashed) domain", y=1.01)
plt.tight_layout()
savefig("roc_source_vs_target")

# Precision-Recall
fig, axs = grid(len(MACHINES))
for ax, mc in zip(axs, MACHINES):
    g = test_all[test_all.machine == mc]
    for m in MODELS:
        p, r, _ = precision_recall_curve(g.label, g[m])
        ax.plot(r, p, color=MODEL_COLORS[m], label=f"{MODEL_NAMES[m]} (AP {average_precision_score(g.label, g[m]):.3f})")
    ax.set_title(mc); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0.4, 1.02); ax.legend(fontsize=7, loc="lower left")
fig.suptitle("Precision-Recall curves", y=1.01)
plt.tight_layout()
savefig("pr_curves")
dbg_end("plot_roc", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 14. Score distributions
# Normal vs anomalous test clips in the source and target domain. The dashed black line is the mean decision threshold (gamma fit on held-out normal source clips). When the *target-normal* distribution sits to the right of the *source-normal* one, normal clips of the shifted domain look more "anomalous" - that is the domain-shift problem.

# %%
_dbg_tok = dbg_start("plot_hist")          # --- debug (auto): banner + timer ---
hdf = test_all.copy()
hdf["group"] = hdf.domain + " " + hdf.label.map({0: "normal", 1: "anomaly"})
hue_order = ["source normal", "target normal", "source anomaly", "target anomaly"]
hist_pal = dict(zip(hue_order, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
thr_by_machine = {mc: fit_thresholds(scores_all[scores_all.machine == mc]) for mc in MACHINES}

for m in MODELS:
    fig, axs = grid(len(MACHINES))
    for ax, mc in zip(axs, MACHINES):
        g = hdf[hdf.machine == mc].copy()
        g["x"] = np.maximum(g[m], 1e-6)
        sns.histplot(data=g, x="x", hue="group", hue_order=hue_order, palette=hist_pal, log_scale=True, bins=35,
                     stat="density", common_norm=False, element="step", fill=False, ax=ax, legend=(mc == MACHINES[0]))
        ax.axvline(np.mean([v for (s, mm), v in thr_by_machine[mc].items() if mm == m]), color="k", ls="--", lw=1)
        ax.set_title(mc); ax.set_xlabel(f"{MODEL_NAMES[m]} anomaly score (log)")
    fig.suptitle(f"Score distributions - {MODEL_NAMES[m]}", y=1.01)
    plt.tight_layout()
    savefig(f"score_hist_{m}")
dbg_end("plot_hist", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 15. Confusion matrices
# Rows are the true class, columns the predicted class; each cell shows the count and the percentage of its row. First the pooled matrices over all machines (per model and per domain), then one matrix per machine and model.

# %%
_dbg_tok = dbg_start("plot_cm")          # --- debug (auto): banner + timer ---
def draw_cm(ax, tn, fp, fn, tp, title, cmap="Blues"):
    cm = np.array([[tn, fp], [fn, tp]])
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    annot = np.array([[f"{cm[i, j]}\n{pct[i, j] * 100:.1f}%" for j in range(2)] for i in range(2)])
    sns.heatmap(pct, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=1, cbar=False, ax=ax,
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"], annot_kws={"fontsize": 10})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)


# pooled over all machines
fig, axs = plt.subplots(len(MODELS), 3, figsize=(12, 3.3 * len(MODELS)))
for r, m in enumerate(MODELS):
    for c, dom in enumerate(["all", "source", "target"]):
        row = cls_tab[(cls_tab.machine == "ALL") & (cls_tab.model == m) & (cls_tab.domain == dom)].iloc[0]
        draw_cm(axs[r, c], row.TN, row.FP, row.FN, row.TP,
                f"{MODEL_NAMES[m]} - {'both domains' if dom == 'all' else dom + ' domain'}\nF1 {row.F1:.3f} | Bal.Acc {row.BalancedAcc:.3f}")
plt.tight_layout()
savefig("confusion_matrix_pooled")

# per machine (both domains)
fig, axs = plt.subplots(len(MACHINES), len(MODELS), figsize=(3.6 * len(MODELS), 3.5 * len(MACHINES)), squeeze=False)
for r, mc in enumerate(MACHINES):
    for c, m in enumerate(MODELS):
        row = cls_tab[(cls_tab.machine == mc) & (cls_tab.model == m) & (cls_tab.domain == "all")].iloc[0]
        draw_cm(axs[r, c], row.TN, row.FP, row.FN, row.TP, f"{mc} - {MODEL_NAMES[m]}\nF1 {row.F1:.3f}")
plt.tight_layout()
savefig("confusion_matrix_per_machine")

# threshold-based metric bars (pooled)
mt = cls_tab[cls_tab.machine == "ALL"].melt(id_vars=["model", "domain"],
                                            value_vars=["Accuracy", "Precision", "Recall", "Specificity", "F1", "BalancedAcc", "MCC"])
mt["model"] = mt.model.map(MODEL_NAMES)
fig, axs = plt.subplots(1, 3, figsize=(22, 4.4), sharey=True)
for ax, dom in zip(axs, ["all", "source", "target"]):
    sns.barplot(data=mt[mt.domain == dom], x="variable", y="value", hue="model", palette=palette, ax=ax)
    ax.set_title({"all": "both domains", "source": "source domain", "target": "target domain"}[dom]); ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    if dom != "all":
        ax.legend_.remove()
    else:
        ax.legend(fontsize=7, loc="lower left")
fig.suptitle("Threshold-based metrics (all machines pooled)", y=1.03)
plt.tight_layout()
savefig("threshold_metrics_pooled")
dbg_end("plot_cm", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 16. Domain shift study: robustness to additive noise
# Extra white or pink (1/f, factory-like) noise is added to the **test** clips at decreasing signal-to-noise ratios. The trained models are not changed. A model that learned *machine-specific structure* should degrade gracefully, a model that only learned "what the training noise looks like" will not.

# %%
_dbg_tok = dbg_start("plot_noise")          # --- debug (auto): banner + timer ---
if noise_all is None or len(noise_all) == 0:
    print("No noise-study results found (CFG['NOISE_SNRS'] is empty).")
else:
    noise_sec = section_table(noise_all, extra=["noise", "snr"])
    noise_tab = machine_table(noise_sec, extra=["noise", "snr"])
    clean = mach_tab.copy()
    clean["snr"] = 30                                   # 30 stands for "clean" on the axis
    frames = []
    for kind in noise_tab.noise.unique():
        frames.append(noise_tab[noise_tab.noise == kind])
        frames.append(clean.assign(noise=kind))
    nplot = pd.concat(frames, ignore_index=True)
    nplot["model_name"] = nplot.model.map(MODEL_NAMES)
    nplot.drop(columns="model_name").to_csv(TAB_DIR / "noise_robustness.csv", index=False)

    summary = (nplot.groupby(["noise", "snr", "model_name"])[["AUC_source", "AUC_target", "pAUC", "official"]].mean().round(4))
    display(summary)

    kinds = list(noise_tab.noise.unique())
    fig, axs = plt.subplots(2, len(kinds), figsize=(6.5 * len(kinds), 8), squeeze=False)
    for c, kind in enumerate(kinds):
        for r, metric in enumerate(["official", "AUC_target"]):
            ax = axs[r, c]
            sns.lineplot(data=nplot[nplot.noise == kind], x="snr", y=metric, hue="model_name", style="model_name",
                         markers=True, dashes=False, palette=palette, errorbar="sd", ax=ax)
            ax.set_xticks(sorted(nplot.snr.unique(), reverse=True))
            ax.set_xticklabels(["clean" if v == 30 else str(v) for v in sorted(nplot.snr.unique(), reverse=True)])
            ax.invert_xaxis()
            ax.set_xlabel("SNR of added noise (dB)")
            ax.set_ylabel({"official": "official score", "AUC_target": "AUC (target domain)"}[metric])
            ax.set_title(f"{kind} noise - {metric} (mean +/- sd over machines)")
            ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("noise_robustness")
dbg_end("plot_noise", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 17. Ablation: which component of DA-ArcNet helps?
# Trained jointly with the main models on `CFG["ABLATION_MACHINES"]` (same data split, same epochs, same seed), so the comparison is controlled. Each row adds exactly one component to the ArcFace-kNN baseline:
# 
# | Variant | AdaCMN | Gradient-reversal domain adversary | AE fusion |
# |---|:-:|:-:|:-:|
# | ArcFace-kNN (baseline) | | | |
# | + AdaCMN | x | | |
# | + DANN | | x | |
# | DA-ArcNet (proposed) | x | x | |
# | DA-ArcNet + AE fusion | x | x | x |

# %%
_dbg_tok = dbg_start("ablation")          # --- debug (auto): banner + timer ---
ABL = ["arc", "arc_ada", "arc_dann", "prop", "full"]
ABL_NAMES = {"arc": "ArcFace-kNN (baseline)", "arc_ada": "+ AdaCMN", "arc_dann": "+ DANN",
             "prop": "DA-ArcNet (AdaCMN + DANN)", "full": "DA-ArcNet + AE fusion"}
abl_machines = [m for m in CFG["ABLATION_MACHINES"] if m in MACHINES]
if not abl_machines:
    print("Ablation skipped (CFG['ABLATION_MACHINES'] is empty).")
    abl_tab = pd.DataFrame()
else:
    sub = test_all[test_all.machine.isin(abl_machines)]
    a_sec = section_table(sub, models=ABL)
    a_mach = machine_table(a_sec)
    a_all = overall_table(a_sec).assign(machine="MEAN")
    abl_tab = pd.concat([a_mach, a_all[a_mach.columns]], ignore_index=True)
    abl_tab["variant"] = abl_tab.model.map(ABL_NAMES)
    abl_tab.to_csv(TAB_DIR / "ablation_results.csv", index=False)
    display(abl_tab[["machine", "variant", "AUC_source", "AUC_target", "pAUC", "official"]].round(4))

    mcs = abl_machines + ["MEAN"]
    fig, axs = plt.subplots(1, len(mcs), figsize=(5.2 * len(mcs), 4.4), squeeze=False)
    for ax, mc in zip(axs[0], mcs):
        g = abl_tab[abl_tab.machine == mc].melt(id_vars="variant", value_vars=["AUC_source", "AUC_target", "pAUC"])
        sns.barplot(data=g, x="variant", y="value", hue="variable", order=[ABL_NAMES[m] for m in ABL], ax=ax)
        ax.set_ylim(0.4, 1.02); ax.set_title(f"Ablation - {mc}"); ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25); ax.axhline(0.5, color="k", ls=":", lw=1)
        if mc != mcs[0]:
            ax.legend_.remove()
    plt.tight_layout()
    savefig("ablation")
dbg_end("ablation", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 18. Cross-dataset evaluation: MIMII DG <-> ToyADMOS2
# Does a representation learned on **one dataset** transfer to the machines of the **other**?
# 
# * **Direction A:** train the embedding on the 5 MIMII DG machines (normal clips only), test on ToyCar and ToyTrain.
# * **Direction B:** train on ToyCar + ToyTrain, test on the 5 MIMII DG machines.
# * The network is **never trained on the test dataset**. Only its *normal training clips* are embedded to build the kNN reference bank (no labels, no anomalies, no fine-tuning); normalisation statistics are fitted per machine on those same normal clips.
# * The pooled models use `ArcFace-kNN` (baseline) and `DA-ArcNet` (proposed). The classes of the ArcFace head are `machine | section | attributes`.
# * Reference line: the **in-dataset** result from Section 10 (model trained on the machine itself).

# %%
_dbg_tok = dbg_start("cross_dataset")          # --- debug (auto): banner + timer ---
XD_FILE = TAB_DIR / "cross_dataset.csv"


def load_pool(machines):
    # normalised features of several machines, pooled; per-machine normalisation (fitted on that machine's training clips)
    parts = []
    for mc in machines:
        meta, X = get_data(mc)
        tr, va, te = split_indices(meta, CFG["SEED"])
        st = fit_norm(X[tr], CFG["NORM_MODE"])
        parts.append(dict(machine=mc, meta=meta, F=apply_norm(X, st), tr=tr, va=va, te=te))
        del X
    return parts


def cross_eval(train_machines, test_machines, tag):
    set_seed(CFG["SEED"])
    pool = load_pool(train_machines)
    T = min(p["F"].shape[1] for p in pool)
    Ftr = np.concatenate([p["F"][p["tr"], :T] for p in pool])
    Fva = np.concatenate([p["F"][p["va"], :T] for p in pool])
    keys = lambda idx_name: np.concatenate([(p["machine"] + "|" + p["meta"].cls_key.iloc[p[idx_name]]).values for p in pool])
    classes = sorted(set(keys("tr")))
    cidx = {k: i for i, k in enumerate(classes)}
    ytr, yva = np.array([cidx[k] for k in keys("tr")]), np.array([cidx[k] for k in keys("va")])
    dtr = np.concatenate([(p["meta"].domain.iloc[p["tr"]] == "target").astype(int).values for p in pool])
    dva = np.concatenate([(p["meta"].domain.iloc[p["va"]] == "target").astype(int).values for p in pool])
    del pool
    print(f"[{tag}] pooled training clips: {len(Ftr)} | classes: {len(classes)}")
    describe(f"[{tag}] pooled training features", Ftr)
    dbg(f"[{tag}] target-domain clips in pool: {int(dtr.sum())} of {len(dtr)}")
    models = {}
    for name, ada, dann in [("arc", False, False), ("prop", True, True)]:
        models[name], _ = train_arc(Ftr, ytr, dtr, Fva, yva, dva, len(classes), CFG, ada, dann,
                                    epochs=CFG["XD_EPOCHS"], seed=CFG["SEED"])
    del Ftr, Fva
    rows = []
    for mc in test_machines:
        meta, X = get_data(mc)
        tr, va, te = split_indices(meta, CFG["SEED"])
        F = apply_norm(X, fit_norm(X[tr], CFG["NORM_MODE"]))
        del X
        sub = meta.iloc[te][["section", "domain", "label"]].reset_index(drop=True)
        for name, m in models.items():
            ref = embed_clips(m, F[tr], CFG)
            sc = knn_scores(embed_clips(m, F[te], CFG), meta.section.values[te], ref, meta.section.values[tr], CFG["KNN_K"])
            d = sub.assign(machine=mc, **{name: sc})
            rows.append((name, d))
        del F
    out = []
    for name in models:
        d = pd.concat([r for n, r in rows if n == name], ignore_index=True)
        t = machine_table(section_table(d, models=[name])).assign(setting=tag)
        out.append(t)
    return pd.concat(out, ignore_index=True)


if not CFG["RUN_CROSS_DATASET"]:
    print("Cross-dataset study skipped (CFG['RUN_CROSS_DATASET'] is False).")
    xd_tab = pd.DataFrame()
elif CFG["RESUME"] and XD_FILE.exists():
    xd_tab = pd.read_csv(XD_FILE)
    print("Loaded saved cross-dataset results.")
else:
    mim = [m for m in CFG["MIMII_MACHINES"] if m in MACHINES]
    toy = [m for m in CFG["TOY_MACHINES"] if m in MACHINES]
    parts = []
    if mim and toy:
        parts.append(cross_eval(mim, toy, "MIMII DG -> ToyADMOS2"))
        tf.keras.backend.clear_session(); gc.collect()
        parts.append(cross_eval(toy, mim, "ToyADMOS2 -> MIMII DG"))
    xd_tab = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    xd_tab.to_csv(XD_FILE, index=False)

if len(xd_tab):
    # in-dataset reference: models trained on the machine itself (Section 10)
    ref = mach_tab[mach_tab.model.isin(["arc", "prop"])].assign(setting="in-dataset (trained on the machine)")
    xd_all = pd.concat([ref[xd_tab.columns], xd_tab], ignore_index=True)
    xd_all["Model"] = xd_all.model.map(MODEL_NAMES)
    xd_show = xd_all.pivot_table(index=["machine", "Model"], columns="setting", values="official").round(4)
    display(xd_show)
    g = xd_all.groupby(["setting", "Model"])[["AUC_source", "AUC_target", "pAUC", "official"]].mean().round(4)
    print("\nMean over test machines:")
    display(g)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.barplot(data=xd_all, x="machine", y="official", hue="setting", ax=ax, errorbar=None,
                order=[m for m in MACHINES if m in xd_all.machine.unique()])
    ax.axhline(0.5, color="k", ls=":", lw=1); ax.set_ylim(0.4, 1.02); ax.set_xlabel("")
    ax.set_ylabel("official score"); ax.set_title("Cross-dataset transfer (bars grouped by training setting; both models averaged)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("cross_dataset")
dbg_end("cross_dataset", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 19. Repeated runs (mean +/- std over seeds)
# A single training run can be lucky. For `CFG["SEED_MACHINES"]` the whole pipeline is repeated with different random seeds (same data split, different initialisation, crop sampling and augmentation). The first seed is the main run.

# %%
_dbg_tok = dbg_start("multi_seed")          # --- debug (auto): banner + timer ---
seed_machines = [m for m in CFG["SEED_MACHINES"] if m in MACHINES]
seed_tab = pd.DataFrame()
if not CFG["RUN_SEEDS"] or not seed_machines or not CFG["EXTRA_SEEDS"]:
    print("Multi-seed study skipped.")
else:
    for mc in seed_machines:
        for sd in CFG["EXTRA_SEEDS"]:
            f = TAB_DIR / f"scores_{mc}_seed{CFG['SEED'] + sd}.csv"   # same name process_machine writes
            if CFG["RESUME"] and f.exists():
                continue
            process_machine(mc, seed=CFG["SEED"] + sd, save="scores", noise=False, ablation=False)
            tf.keras.backend.clear_session(); gc.collect()
    rows = []
    for mc in seed_machines:
        runs = [(CFG["SEED"], scores_all[(scores_all.machine == mc) & (scores_all.split == "test")])]
        for sd in CFG["EXTRA_SEEDS"]:
            f = TAB_DIR / f"scores_{mc}_seed{CFG['SEED'] + sd}.csv"
            if f.exists():
                d = pd.read_csv(f)
                runs.append((CFG["SEED"] + sd, d[d.split == "test"]))
        for seed_v, d in runs:
            t = machine_table(section_table(d.reset_index(drop=True)))
            rows.append(t.assign(seed=seed_v))
    seed_tab = pd.concat(rows, ignore_index=True)
    seed_tab.to_csv(TAB_DIR / "multi_seed.csv", index=False)
    dbg(f"multi-seed: runs found per machine = {seed_tab.groupby('machine').seed.nunique().to_dict()}")
    agg = seed_tab.groupby(["machine", "model"])[["AUC_source", "AUC_target", "pAUC", "official"]].agg(["mean", "std"])
    n_seeds = seed_tab.groupby("machine").seed.nunique().to_dict()
    print("Number of runs per machine:", n_seeds)
    out = (agg["official"]["mean"].map("{:.3f}".format) + " +/- " + agg["official"]["std"].fillna(0).map("{:.3f}".format)).unstack("model")[MODELS]
    out.columns = [MODEL_NAMES[c] for c in out.columns]
    print("\nOfficial score, mean +/- std over seeds")
    display(out)
    out2 = (agg["AUC_target"]["mean"].map("{:.3f}".format) + " +/- " + agg["AUC_target"]["std"].fillna(0).map("{:.3f}".format)).unstack("model")[MODELS]
    out2.columns = [MODEL_NAMES[c] for c in out2.columns]
    print("\nTarget-domain AUC, mean +/- std over seeds")
    display(out2)
dbg_end("multi_seed", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 20. Explainability and error analysis
# Three views of *why* the models decide as they do:
# 
# 1. **Autoencoder residual map** - the squared reconstruction error in time x frequency: bright regions are what the autoencoder could not reproduce.
# 2. **Grad-CAM for DA-ArcNet** - the gradient of the "distance to the nearest normal reference clip" (the anomaly score) with respect to the last convolutional feature map, shown over the spectrogram.
# 3. **Embedding space (t-SNE)** and a **domain-probe**: a logistic-regression classifier tries to tell source from target normal clips using the embedding. Lower accuracy means the embedding is *more domain-invariant*.
# 
# Caution: Grad-CAM and residual maps show where the model looks, not proof of a physical fault location.

# %%
_dbg_tok = dbg_start("explain")          # --- debug (auto): banner + timer ---
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

probe_tab = pd.DataFrame()
xai_machine = ("fan" if "fan" in MACHINES else MACHINES[0])
xai_dir = FIG_DIR


def load_machine_models(machine):
    ae = tf.keras.models.load_model(MODEL_DIR / f"{machine}_autoencoder.keras")
    prop = tf.keras.models.load_model(MODEL_DIR / f"{machine}_proposed_dannet.keras")
    nz = np.load(MODEL_DIR / f"{machine}_norm.npz")
    st = dict(mode=str(nz["mode"]), mu=nz["mu"], sd=nz["sd"])
    rf = np.load(MODEL_DIR / f"{machine}_refs.npz")
    return ae, prop, st, rf


_GM_CACHE = {}


def gradcam_clip(prop, F1, ref_sec, crop=64, stride=32):
    # Grad-CAM map [T, M] of one clip F1 [T, M] for the anomaly score (1 - max cosine to the same-section normal references)
    if id(prop) not in _GM_CACHE:
        _GM_CACHE[id(prop)] = tf.keras.Model(prop.input, [prop.get_layer("last_conv").output, prop.output])
    gm = _GM_CACHE[id(prop)]
    T, M = F1.shape
    acc, cnt = np.zeros((T, M)), np.zeros((T, 1))
    R = tf.constant(ref_sec.reshape(-1, ref_sec.shape[-1]), tf.float32)
    for s in range(0, T - crop + 1, stride):
        x = tf.constant(F1[s:s + crop][None, ..., None], tf.float32)
        with tf.GradientTape() as tape:
            A, e = gm(x, training=False)
            sim = tf.matmul(e, R, transpose_b=True)
            score = 1.0 - tf.reduce_max(sim)
        g = tape.gradient(score, A)
        w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(w * A, axis=-1, keepdims=True))
        cam = tf.image.resize(cam, (crop, M)).numpy()[0, :, :, 0]
        acc[s:s + crop] += cam
        cnt[s:s + crop] += 1
    cam = acc / np.maximum(cnt, 1)
    return cam / (cam.max() + 1e-9)


try:
    ae_x, prop_x, st_x, rf_x = load_machine_models(xai_machine)
    meta_x, X_x = get_data(xai_machine)
    F_x = apply_norm(X_x, st_x)
    sc = scores_all[(scores_all.machine == xai_machine) & (scores_all.split == "test")]
    # choose, for section 0: a normal source, a normal target, an anomaly source, an anomaly target clip (median-score ones)
    picks = []
    for dom in ("source", "target"):
        for lab in (0, 1):
            g = sc[(sc.section == 0) & (sc.domain == dom) & (sc.label == lab)].sort_values("prop")
            if len(g):
                picks.append((dom, lab, g.iloc[len(g) // 2]["name"]))
    fig, axs = plt.subplots(3, len(picks), figsize=(5 * len(picks), 9), squeeze=False)
    for c, (dom, lab, fname) in enumerate(picks):
        i = int(meta_x.index[meta_x["name"] == fname][0])
        lm, Fi = X_x[i], F_x[i]
        res = ae_residual_map(ae_x, Fi, CFG)
        ref_sec = rf_x["prop"][rf_x["sections"] == 0]
        cam = gradcam_clip(prop_x, Fi, ref_sec)
        ttl = f"{dom} {'ANOMALY' if lab else 'normal'}"
        axs[0, c].imshow(lm.T, origin="lower", aspect="auto", cmap="magma"); axs[0, c].set_title(f"log-mel - {ttl}")
        axs[1, c].imshow(res.T, origin="lower", aspect="auto", cmap="inferno"); axs[1, c].set_title("AE residual map")
        axs[2, c].imshow(lm.T, origin="lower", aspect="auto", cmap="gray")
        axs[2, c].imshow(cam.T, origin="lower", aspect="auto", cmap="jet", alpha=0.45); axs[2, c].set_title("DA-ArcNet Grad-CAM")
        for r in range(3):
            axs[r, c].set_xlabel("frame"); axs[r, c].set_ylabel("mel bin")
    fig.suptitle(f"{xai_machine}, section 00: input, autoencoder residual and Grad-CAM (median-scored clip per group)", y=1.01)
    plt.tight_layout()
    savefig("explainability_maps")

    # which frequency bands drive the anomaly score? mean Grad-CAM per mel band, normal vs anomaly (section 0, test)
    bands = {}
    for lab in (0, 1):
        names = sc[(sc.section == 0) & (sc.label == lab)]["name"].head(12)
        cams = [gradcam_clip(prop_x, F_x[int(meta_x.index[meta_x["name"] == n][0])], rf_x["prop"][rf_x["sections"] == 0]).mean(axis=0) for n in names]
        bands[lab] = np.mean(cams, axis=0)
    plt.figure(figsize=(9, 3.4))
    plt.plot(bands[0], label="normal clips"); plt.plot(bands[1], label="anomalous clips")
    plt.xlabel("mel bin"); plt.ylabel("mean Grad-CAM"); plt.legend()
    plt.title(f"{xai_machine}: frequency bands that drive the anomaly score")
    savefig("gradcam_bands")
    del X_x, F_x
except Exception as e:
    print("Explainability maps skipped:", e)

# ---- embedding space + domain probe
try:
    z = np.load(TAB_DIR / f"emb_{xai_machine}.npz", allow_pickle=True)
    sec_m = (z["section"] == 0)
    dom_a = z["domain"][sec_m]; lab_a = z["label"][sec_m]; spl_a = z["split"][sec_m]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
    probe_rows = []
    for ax, key, name in zip(axs, ["emb_arc", "emb_prop"], ["ArcFace-kNN (baseline)", "DA-ArcNet (proposed)"]):
        E = z[key][sec_m]
        Y = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(E)
        for (d, l), col, mk in zip([("source", 0), ("target", 0), ("source", 1), ("target", 1)],
                                   ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], ["o", "o", "x", "x"]):
            m = (dom_a == d) & (lab_a == l)
            ax.scatter(Y[m, 0], Y[m, 1], s=14, c=col, marker=mk, alpha=0.65, label=f"{d} {'anomaly' if l else 'normal'}")
        ax.set_title(f"t-SNE - {name}"); ax.set_xticks([]); ax.set_yticks([])
        # domain probe on NORMAL clips only
        nm = (lab_a == 0)
        yd = (dom_a[nm] == "target").astype(int)
        if yd.sum() >= 5:
            pred = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), E[nm], yd,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=0))
            probe_rows.append(dict(embedding=name, domain_probe_balanced_acc=balanced_accuracy_score(yd, pred)))
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    savefig("embedding_tsne")
    probe_tab = pd.DataFrame(probe_rows)
    probe_tab.to_csv(TAB_DIR / "domain_probe.csv", index=False)
    print("Domain probe (normal clips, section 0): can a linear classifier tell source from target?  0.5 = cannot (domain-invariant)")
    display(probe_tab.round(3))
except Exception as e:
    print("Embedding analysis skipped:", e)

# ---- error-case analysis (proposed + AE fusion at the fixed threshold)
err_m = "full"
et = pred_test.copy()
et["pred"] = et[f"pred_{err_m}"].astype(bool)
et["error"] = np.where((et.label == 0) & et.pred, "false alarm", np.where((et.label == 1) & ~et.pred, "missed anomaly", "correct"))
rate = (et.groupby(["machine", "domain"]).apply(lambda g: pd.Series({
    "false_alarm_rate": ((g.label == 0) & g.pred).sum() / max((g.label == 0).sum(), 1),
    "miss_rate": ((g.label == 1) & ~g.pred).sum() / max((g.label == 1).sum(), 1)})).reset_index())
rate.to_csv(TAB_DIR / "error_rates_by_domain.csv", index=False)
display(rate.round(3))
fig, axs = plt.subplots(1, 2, figsize=(14, 4))
for ax, col, ttl in zip(axs, ["false_alarm_rate", "miss_rate"], ["False-alarm rate on normal clips", "Miss rate on anomalous clips"]):
    sns.barplot(data=rate, x="machine", y=col, hue="domain", order=MACHINES, ax=ax)
    ax.set_title(f"{ttl} ({MODEL_NAMES[err_m]})"); ax.set_xlabel("")
plt.tight_layout()
savefig("error_analysis")

worst_fa = et[et.error == "false alarm"].sort_values(err_m, ascending=False).head(10)[["machine", "name", "domain", err_m]]
worst_miss = et[et.error == "missed anomaly"].sort_values(err_m).head(10)[["machine", "name", "domain", err_m]]
print("Highest-scoring normal clips that raised a false alarm:"); display(worst_fa)
print("Lowest-scoring anomalous clips that were missed:"); display(worst_miss)
worst_fa.to_csv(TAB_DIR / "top_false_alarms.csv", index=False)
worst_miss.to_csv(TAB_DIR / "top_missed_anomalies.csv", index=False)
dbg_end("explain", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 21. Model size and speed
# Parameters and inference time per 10-second clip (feature-to-score, kNN included, measured on the Colab GPU). A lightweight model is the point of the design, so this must be reported next to accuracy.

# %%
_dbg_tok = dbg_start("complexity")          # --- debug (auto): banner + timer ---
rows = []
for mc in MACHINES:
    cf = TAB_DIR / f"complexity_{mc}.json"
    if cf.exists():
        c = json.load(open(cf))
        for m, p in c["params"].items():
            rows.append(dict(machine=mc, model=m, params=p, ms_per_clip=1000 * c["sec_per_clip"][m]))
cx = pd.DataFrame(rows)
if len(cx):
    cx_m = cx.groupby("model")[["params", "ms_per_clip"]].mean()
    cx_m["official"] = over_tab.set_index("model").official.reindex(cx_m.index)
    cx_m = cx_m.loc[[m for m in ["ae", "mnv2", "arc", "arc_ada", "arc_dann", "prop"] if m in cx_m.index]]
    cx_m.insert(0, "name", [MODEL_NAMES[m] for m in cx_m.index])
    cx_m["params"] = cx_m.params.round(0).astype(int)
    cx_m.to_csv(TAB_DIR / "complexity_summary.csv")
    display(cx_m.round(3))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m, r in cx_m.iterrows():
        if m in MODELS:
            ax.scatter(r.params / 1e3, r.official, s=90, color=MODEL_COLORS[m])
            ax.annotate(MODEL_NAMES[m], (r.params / 1e3, r.official), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("parameters (thousands, log scale)"); ax.set_ylabel("official score (mean over machines)")
    ax.set_title("Accuracy vs. model size")
    plt.tight_layout()
    savefig("accuracy_vs_size")
else:
    print("No complexity files found.")
dbg_end("complexity", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 22. Limitations and future work
# 
# **Limitations (be ready to state these in the viva)**
# * **Fixed threshold from source data only.** The decision threshold is fitted on held-out *source* normal clips; the 10 target-domain training clips per section are too few to calibrate one. Threshold-based metrics (F1, accuracy, confusion matrices) are therefore weaker on the target domain than the threshold-free AUC / pAUC.
# * **Ten target-domain clips.** The domain adversary and the oversampling rely on very few target clips; results can vary between sections and seeds (see Section 19).
# * **ToyADMOS2 coverage.** Only ToyCar and ToyTrain, taken from the DCASE 2022 development set, represent ToyADMOS2. The full 169 GB raw record was not used.
# * **Reimplemented baselines.** The three baselines are re-implementations in the style of the cited systems with the same budget as the proposed model, *not* the authors' official code or challenge-winning ensembles; absolute numbers are not directly comparable with the DCASE leaderboards.
# * **Compute budget.** Few seeds (3 on two machines), moderate bootstrap size; no hyper-parameter search beyond the values listed in the configuration cell.
# * **Explainability is qualitative.** Grad-CAM and residual maps show where the models respond; they were not validated against ground-truth fault locations.
# * **Synthetic anomalies in ToyADMOS2 / seeded faults in MIMII DG** are not field data; real deployments have additional drift and unseen failure modes.
# 
# **Future work**
# * Calibrate thresholds per domain with a small number of target-domain normal clips, or use domain-specific score normalisation.
# * Combine the lightweight encoder with a pre-trained audio foundation model or LoRA adapters [refs. in Section 0] and distil back to the small network.
# * Use outlier exposure with sounds of *other* machines as pseudo-anomalies.
# * Evaluate on the DCASE 2023-2025 additional-attribute and first-shot settings, and with true field recordings.
# * Quantise the model (INT8 / TFLite) and measure on an edge device.

# %% [markdown]
# ## 23. Paper assets (for the IEEE conference-format paper)
# This cell writes, from the real results, everything the LaTeX paper (`paper/main.tex` in the repository) reads: tables (`generated/tab_*.tex`), numbers (`generated/numbers.tex`), auto-worded findings (`generated/findings.tex`) and the figures. Copy the folder `paper/` from Drive into the repository, compile `main.tex`, and **read the findings text critically** - it only states what the numbers say.

# %%
_dbg_tok = dbg_start("paper_assets")          # --- debug (auto): banner + timer ---
PAPER_DIR = OUT_DIR / "paper"
GEN_DIR, PFIG_DIR = PAPER_DIR / "generated", PAPER_DIR / "figures"
for d in (GEN_DIR, PFIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
SHORT = {"ae": "AE", "mnv2": "MNv2", "arc": "ArcFace-kNN", "arc_ada": "+AdaCMN", "arc_dann": "+DANN",
         "prop": "DA-ArcNet", "full": "DA-ArcNet+AE"}


def write(name, text):
    (GEN_DIR / name).write_text(text)


def macro(name, value):
    return "\\newcommand{\\%s}{%s}\n" % (name, value)


# ---------------- tables ----------------
w = mach_tab.pivot(index="machine", columns="model", values="official")[MODELS].reindex(MACHINES)
w.loc["Mean"] = w.mean()
write("tab_main.tex", latex_table(w.rename(columns=SHORT), "Official score (harmonic mean of source AUC, target AUC and pAUC) per machine. Best per row in bold.",
                                  "tab:main", index_name="Machine"))

ov = over_tab.set_index("model").loc[MODELS]
cio = ci_tab[(ci_tab.machine == "ALL")].set_index(["model", "metric"])
dom = pd.DataFrame(index=[SHORT[m] for m in MODELS])
dom["AUC src"] = [f"{ov.loc[m, 'AUC_source']:.3f}" for m in MODELS]
dom["AUC tgt [95% CI]"] = [f"{ov.loc[m, 'AUC_target']:.3f} [{cio.loc[(m, 'AUC_target'), 'lo']:.3f}, {cio.loc[(m, 'AUC_target'), 'hi']:.3f}]" for m in MODELS]
dom["pAUC"] = [f"{ov.loc[m, 'pAUC']:.3f}" for m in MODELS]
dom["Official [95% CI]"] = [f"{ov.loc[m, 'official']:.3f} [{cio.loc[(m, 'official'), 'lo']:.3f}, {cio.loc[(m, 'official'), 'hi']:.3f}]" for m in MODELS]
write("tab_domain.tex", latex_table(dom, "Mean over all machines and sections; 95\\% bootstrap confidence intervals for target AUC and official score.",
                                    "tab:domain", bold=None, index_name="Model", col_spec="lcccc"))

pooled = cls_tab[(cls_tab.machine == "ALL") & (cls_tab.domain == "all")].set_index("model").loc[MODELS]
thr = pooled[["Accuracy", "Precision", "Recall", "Specificity", "F1", "MCC"]].copy()
thr.index = [SHORT[m] for m in MODELS]
write("tab_threshold.tex", latex_table(thr, "Threshold-based metrics, all machines and both domains pooled (gamma threshold from held-out normal source clips).",
                                       "tab:thr", bold="col_max", index_name="Model"))

if len(abl_tab):
    a = abl_tab[abl_tab.machine == "MEAN"].set_index("model").loc[ABL][["AUC_source", "AUC_target", "pAUC", "official"]]
    a.index = [ABL_NAMES[m] for m in ABL]; a.columns = ["AUC src", "AUC tgt", "pAUC", "Official"]
    write("tab_ablation.tex", latex_table(a, "Ablation on " + ", ".join(abl_machines) + " (mean over these machines).",
                                          "tab:abl", bold="col_max", index_name="Variant"))
if len(xd_tab):
    x = xd_all.groupby(["setting", "Model"])["official"].mean().unstack("setting")
    x.index = [i.replace("[DCASE22]", "").replace("[SOTA-style]", "").strip() for i in x.index]
    x.columns = [{"MIMII DG -> ToyADMOS2": "MIMII to Toy", "ToyADMOS2 -> MIMII DG": "Toy to MIMII"}.get(c, "In-dataset" if str(c).startswith("in-dataset") else c) for c in x.columns]
    write("tab_crossdata.tex", latex_table(x, "Cross-dataset transfer, official score averaged over the test machines.",
                                           "tab:xd", bold="row_max", index_name="Model", col_spec="lccc"))
if len(seed_tab):
    sg = seed_tab.groupby(["machine", "model"]).official.agg(["mean", "std"]).reset_index()
    sw = sg.assign(v=sg["mean"].map("{:.3f}".format) + " $\\pm$ " + sg["std"].fillna(0).map("{:.3f}".format)).pivot(index="machine", columns="model", values="v")[MODELS]
    sw.columns = [SHORT[c] for c in sw.columns]
    write("tab_seeds.tex", latex_table(sw, "Official score, mean $\\pm$ std over %d seeds." % seed_tab.seed.nunique(), "tab:seeds", bold=None, index_name="Machine"))
if noise_all is not None and len(noise_all):
    ns = section_table(noise_all, extra=["noise", "snr"]); nt = machine_table(ns, extra=["noise", "snr"])
    nt = pd.concat([nt, mach_tab.assign(snr=99, noise="clean")], ignore_index=True)
    nm = nt.groupby(["model", "snr"]).official.mean().unstack("snr").loc[MODELS]
    nm.columns = ["clean" if c == 99 else f"{int(c)} dB" for c in nm.columns]
    nm.index = [SHORT[m] for m in MODELS]
    write("tab_noise.tex", latex_table(nm, "Official score under additive noise (white and pink averaged), mean over machines.", "tab:noise",
                                       bold="col_max", index_name="Model"))
try:
    cxs = pd.read_csv(TAB_DIR / "complexity_summary.csv", index_col=0)
    ct = cxs[["params", "ms_per_clip", "official"]].copy()
    ct.index = [SHORT[m] for m in ct.index]; ct.columns = ["Parameters", "ms / clip", "Official"]
    ct["Parameters"] = ct["Parameters"].map("{:,.0f}".format).str.replace(",", "\\,")
    ct["ms / clip"] = ct["ms / clip"].map("{:.1f}".format); ct["Official"] = ct["Official"].map(lambda v: "--" if pd.isna(v) else f"{v:.3f}")
    write("tab_complexity.tex", latex_table(ct, "Model size and inference time per 10-s clip.", "tab:cx", bold=None, index_name="Model"))
except Exception as e:
    print("complexity table skipped:", e)

wt = wil_tab[(wil_tab.metric == "official")].copy()
wt["cand"] = wt.candidate.map(SHORT); wt["base"] = wt.baseline.map(SHORT)
wtt = wt.pivot(index="cand", columns="base", values="p_greater")
write("tab_wilcoxon.tex", latex_table(wtt, "One-sided Wilcoxon signed-rank test over machine-sections (official score): p-value that the row model beats the column baseline.",
                                      "tab:wil", fmt="{:.3f}", bold="row_min", index_name="Candidate"))

# ---------------- numbers and findings ----------------
base_keys = ["ae", "mnv2", "arc"]
best_b = ov.loc[base_keys].official.idxmax()
gain = ov.loc["prop", "official"] - ov.loc[best_b, "official"]
gain_full = ov.loc["full", "official"] - ov.loc[best_b, "official"]
gain_tgt = ov.loc["prop", "AUC_target"] - ov.loc[best_b, "AUC_target"]
p_best = wil_tab[(wil_tab.candidate == "prop") & (wil_tab.baseline == best_b) & (wil_tab.metric == "official")].p_greater.iloc[0]
lo_p, hi_p = cio.loc[("prop", "official"), "lo"], cio.loc[("prop", "official"), "hi"]
lo_b, hi_b = cio.loc[(best_b, "official"), "lo"], cio.loc[(best_b, "official"), "hi"]
overlap = not (lo_p > hi_b or lo_b > hi_p)
gap = {m: ov.loc[m, "AUC_source"] - ov.loc[m, "AUC_target"] for m in MODELS}

nums = ""
for m in MODELS:
    nums += macro("Off" + {"ae": "AE", "mnv2": "MN", "arc": "Arc", "prop": "Prop", "full": "Full"}[m], f"{ov.loc[m, 'official']:.3f}")
    nums += macro("Tgt" + {"ae": "AE", "mnv2": "MN", "arc": "Arc", "prop": "Prop", "full": "Full"}[m], f"{ov.loc[m, 'AUC_target']:.3f}")
    nums += macro("Src" + {"ae": "AE", "mnv2": "MN", "arc": "Arc", "prop": "Prop", "full": "Full"}[m], f"{ov.loc[m, 'AUC_source']:.3f}")
nums += macro("OffBestBase", f"{ov.loc[best_b, 'official']:.3f}") + macro("BestBaseline", SHORT[best_b]) + macro("GainOverBest", f"{gain:+.3f}") + macro("GainFullOverBest", f"{gain_full:+.3f}")
nums += macro("GainTargetOverBest", f"{gain_tgt:+.3f}") + macro("PWilcoxon", f"{p_best:.3f}")
nums += macro("NMachines", str(len(MACHINES))) + macro("NBoot", str(CFG["BOOT_N"])) + macro("NSeeds", str(1 + len(CFG["EXTRA_SEEDS"]) if CFG["RUN_SEEDS"] else 1))
if len(abl_tab):
    ab = abl_tab[abl_tab.machine == "MEAN"].set_index("model").official
    nums += macro("AblArc", f"{ab['arc']:.3f}") + macro("AblAda", f"{ab['arc_ada']:.3f}") + macro("AblDann", f"{ab['arc_dann']:.3f}") + macro("AblProp", f"{ab['prop']:.3f}")
if len(probe_tab) == 2:
    nums += macro("ProbeArc", f"{probe_tab.domain_probe_balanced_acc.iloc[0]:.2f}") + macro("ProbeProp", f"{probe_tab.domain_probe_balanced_acc.iloc[1]:.2f}")
write("numbers.tex", nums)

F = []
F.append(f"Averaged over {len(MACHINES)} machines, the proposed DA-ArcNet reaches an official score of {ov.loc['prop','official']:.3f}, "
         f"compared with {ov.loc[best_b,'official']:.3f} for the strongest baseline ({SHORT[best_b]}), a difference of {gain:+.3f}.")
if overlap:
    F.append(f"The 95\\% bootstrap intervals of the two overlap ([{lo_p:.3f}, {hi_p:.3f}] versus [{lo_b:.3f}, {hi_b:.3f}]), so this difference "
             f"should be read as indicative rather than conclusive.")
else:
    F.append(f"The 95\\% bootstrap intervals do not overlap ([{lo_p:.3f}, {hi_p:.3f}] versus [{lo_b:.3f}, {hi_b:.3f}]).")
F.append(f"A one-sided Wilcoxon signed-rank test over machine-sections gives $p={p_best:.3f}$ for DA-ArcNet against {SHORT[best_b]}.")
F.append(f"On the shifted target domain the AUC of DA-ArcNet is {ov.loc['prop','AUC_target']:.3f} ({gain_tgt:+.3f} relative to {SHORT[best_b]}); "
         f"the source-to-target AUC drop is {gap['prop']:+.3f} for DA-ArcNet and {gap['arc']:+.3f} for ArcFace-kNN.")
F.append(f"Fusing the autoencoder score gives {ov.loc['full','official']:.3f} ({gain_full:+.3f} relative to {SHORT[best_b]}).")
if len(abl_tab):
    F.append(f"In the ablation on {', '.join(abl_machines)}, the official score moves from {ab['arc']:.3f} (ArcFace-kNN) to {ab['arc_ada']:.3f} with AdaCMN, "
             f"{ab['arc_dann']:.3f} with the domain adversary and {ab['prop']:.3f} with both.")
if len(probe_tab) == 2:
    F.append(f"A linear domain probe separates source from target normal clips with balanced accuracy {probe_tab.domain_probe_balanced_acc.iloc[0]:.2f} "
             f"on the ArcFace-kNN embedding and {probe_tab.domain_probe_balanced_acc.iloc[1]:.2f} on the DA-ArcNet embedding (0.5 = domain-invariant).")
write("findings.tex", "\n".join(F) + "\n")
print("\n".join(x.replace("\\%", "%").replace("$", "") for x in F))

# ---------------- figures ----------------
wanted = ["architecture", "input_pipeline_1", "metrics_per_machine", "domain_shift_gap", "roc_curves", "confusion_matrix_pooled",
          "noise_robustness", "ablation", "cross_dataset", "explainability_maps", "embedding_tsne", "error_analysis", "accuracy_vs_size",
          "eda_pca_domains", "training_curves"]
copied = []
for n in wanted:
    p = FIG_DIR / f"{n}.png"
    if p.exists():
        shutil.copy(p, PFIG_DIR / p.name)
        copied.append(p.name)
print("\nPaper assets written to", PAPER_DIR, "| figures:", len(copied))
dbg(f"paper assets: tables/macros written = {sorted(p.name for p in GEN_DIR.glob('*'))}")
dbg(f"paper assets: figures missing (run the cells that create them): {sorted(set(wanted) - {c[:-4] for c in copied})}")
dbg_end("paper_assets", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 24. Presentation (slides built from your results)
# Builds `ASD_24I461_presentation.pptx` with the real tables and figures of this run (python-pptx). Open it, **replace nothing silently**: check each number against the tables above, add your own words in the speaker notes, and rehearse the demo (Section 25).

# %%
_dbg_tok = dbg_start("presentation")          # --- debug (auto): banner + timer ---
try:
    from pptx import Presentation
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-pptx"], check=False)
    from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from PIL import Image

DARK, ACC, GREY = RGBColor(0x1F, 0x2A, 0x44), RGBColor(0xC7, 0x4B, 0x2D), RGBColor(0x55, 0x55, 0x55)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def new_slide(title, notes=""):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.95))
    bar.fill.solid(); bar.fill.fore_color.rgb = DARK; bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.12), Inches(12.3), Inches(0.75))
    p = tb.text_frame.paragraphs[0]; p.text = title; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)
    ft = s.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.35))
    q = ft.text_frame.paragraphs[0]; q.text = "Ardhanarieswarar (24I461), Gokul Kannan (24I463), Jafer Sathik B (24I464), Madhan M (24I466) | 23I006 Deep Learning Mini Project | PSG College of Technology, Department of Information Technology"
    q.font.size = Pt(10); q.font.color.rgb = GREY
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def bullets(s, items, left=0.6, top=1.25, width=12.0, height=5.6, size=20):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf_ = tb.text_frame; tf_.word_wrap = True
    for i, it in enumerate(items):
        lvl = 1 if it.startswith("  ") else 0
        p = tf_.paragraphs[0] if i == 0 else tf_.add_paragraph()
        p.text = ("- " if lvl == 0 else "o ") + it.strip(); p.level = lvl
        p.font.size = Pt(size - 3 * lvl); p.space_after = Pt(6)
    return tb


def picture(s, path, left, top, width, height):
    path = Path(path)
    if not path.exists():
        return
    w, h = Image.open(path).size
    sc = min(width / w, height / h)
    s.shapes.add_picture(str(path), Inches(left + (width - w * sc) / 2), Inches(top + (height - h * sc) / 2), Inches(w * sc), Inches(h * sc))


def table(s, df, left, top, width, height, size=12):
    rows, cols = df.shape[0] + 1, df.shape[1] + 1
    t = s.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height)).table
    for j, c in enumerate([df.index.name or ""] + list(df.columns)):
        t.cell(0, j).text = str(c)
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        t.cell(i, 0).text = str(idx)
        for j, v in enumerate(row, start=1):
            t.cell(i, j).text = f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    for r in range(rows):
        for c in range(cols):
            for p in t.cell(r, c).text_frame.paragraphs:
                p.font.size = Pt(size)


# 1 title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = DARK; bg.line.fill.background()
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(3.5)); tf_ = tb.text_frame; tf_.word_wrap = True
p = tf_.paragraphs[0]; p.text = "DA-ArcNet: Lightweight Domain-Adversarial Angular-Margin Learning for Unsupervised Anomalous Sound Detection under Domain Shift"
p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)
for line in ["Ardhanarieswarar (24I461)  |  Gokul Kannan (24I463)", "Jafer Sathik B (24I464)  |  Madhan M (24I466)", "23I006 Deep Learning Mini Project", "PSG College of Technology, Department of Information Technology"]:
    p = tf_.add_paragraph(); p.text = line; p.font.size = Pt(20); p.font.color.rgb = RGBColor(220, 220, 220)

s = new_slide("Problem and motivation", "Explain why only normal data is available and why domain shift breaks naive detectors.")
bullets(s, ["Machines are monitored by sound; faults are rare and varied -> train on NORMAL sounds only (unsupervised ASD)",
            "Domain shift: speed, load, microphone and background noise differ between training (source) and deployment (target)",
            "Benchmark (DCASE 2022 Task 2): per section 990 source + only 10 target normal clips; test = 50 normal + 50 anomalous per domain",
            "Metrics: AUC (source, target), partial AUC (FPR <= 0.1), official score = harmonic mean",
            "Goal: a LIGHTWEIGHT model that stays reliable on the shifted target domain"])

s = new_slide("Literature survey and research gap", "18 verified papers 2022-2026; the full table is in the notebook (Section 0) and paper.")
bullets(s, ["18 papers (2022-2026): DCASE challenge reports, MIMII DG, STgram, Gen-Rep, FeatEx, AdaProj, LoRA adapters, SSAM, diffusion-based UD-ASD ...",
            "Gap 1: top systems are heavy (pre-trained backbones, LoRA, ensembles) -> costly at the edge",
            "Gap 2: domain handling is implicit (loss design, memory banks) - no explicit domain-invariance in an angular-margin embedding",
            "Gap 3: evaluation within one dataset; cross-dataset transfer and noise robustness rarely reported",
            "Gap 4: little explainability or error analysis"])

s = new_slide("Datasets and pre-processing", "Both datasets, same file format; log-mel with train-only normalisation; no test clip is used for fitting anything.")
bullets(s, ["MIMII DG: fan, gearbox, bearing, slider, valve | ToyADMOS2 (ToyCar, ToyTrain via DCASE 2022 dev set)",
            "Log-mel: 16 kHz, FFT 1024, hop 512, 64 mel bands (dB); global z-norm from TRAIN clips only",
            "Hold-out: 10 % of source normal clips for threshold / early stopping / fusion scaling / (not kNN references)",
            "Noise study: white + pink noise at 20/10/5/0 dB added to test clips only"], width=5.6, size=17)
picture(s, FIG_DIR / "input_pipeline_1.png", 6.3, 1.3, 6.7, 5.5)

s = new_slide("Proposed model: DA-ArcNet", "Walk through training path and inference path of the diagram; point at the orange parts as the new contributions.")
picture(s, FIG_DIR / "architecture.png", 0.3, 1.1, 12.7, 4.6)
bullets(s, ["AdaCMN: learnable per-mel blend between identity and clip-mean normalisation", "Gradient-reversal domain adversary + ArcFace condition classifier, target oversampled x10; kNN scoring; optional AE fusion"],
        top=5.8, height=1.2, size=16)

s = new_slide("Experimental protocol", "Baselines are re-implementations with the same budget; say so.")
bullets(s, ["Baselines: (1) DCASE22 AE, (2) DCASE22 MobileNetV2 section classifier, (3) ArcFace-kNN (recent top-system paradigm)",
            "Metrics: AUC source / target, pAUC, official score, EER, AP; Acc / Prec / Recall / Spec / F1 / MCC and confusion matrices at a gamma threshold",
            "Statistics: 95 % bootstrap CIs, multi-seed repeats, paired Wilcoxon tests",
            "Extras: cross-dataset transfer, ablation, noise robustness, Grad-CAM / residual maps, t-SNE + domain probe, size and speed"])

s = new_slide("Main results")
table(s, w.rename(columns=SHORT).round(3), 0.6, 1.3, 12.0, 4.5, size=13)
bullets(s, [f"Official score (mean over machines): DA-ArcNet {ov.loc['prop','official']:.3f} | best baseline {SHORT[best_b]} {ov.loc[best_b,'official']:.3f} | DA-ArcNet+AE {ov.loc['full','official']:.3f}"],
        top=6.0, height=0.9, size=16)

s = new_slide("Domain shift: source vs target")
picture(s, FIG_DIR / "metrics_per_machine.png", 0.3, 1.1, 7.6, 5.8)
picture(s, FIG_DIR / "domain_shift_gap.png", 8.0, 1.3, 5.0, 3.2)
bullets(s, [f"Source-to-target AUC drop: DA-ArcNet {gap['prop']:+.3f}, ArcFace-kNN {gap['arc']:+.3f}"], left=8.0, top=4.7, width=5.0, height=1.5, size=15)

s = new_slide("Statistical rigour")
table(s, dom, 0.5, 1.3, 12.3, 3.0, size=12)
bullets(s, [f"Wilcoxon (official score, DA-ArcNet vs {SHORT[best_b]}): p = {p_best:.3f}",
            "CIs: bootstrap over test clips; seeds: see notebook Section 19" + (" (mean +/- std in paper Table)" if len(seed_tab) else " (not run)")],
        top=4.6, height=1.8, size=17)

s = new_slide("Ablation")
picture(s, FIG_DIR / "ablation.png", 0.4, 1.2, 12.5, 4.4)
if len(abl_tab):
    bullets(s, [f"Mean official score: ArcFace-kNN {ab['arc']:.3f} -> +AdaCMN {ab['arc_ada']:.3f} | +DANN {ab['arc_dann']:.3f} | both {ab['prop']:.3f}"], top=5.9, height=1.0, size=17)

s = new_slide("Cross-dataset evaluation and noise robustness")
picture(s, FIG_DIR / "cross_dataset.png", 0.3, 1.2, 6.4, 5.0)
picture(s, FIG_DIR / "noise_robustness.png", 6.8, 1.2, 6.3, 5.0)

s = new_slide("Explainability")
picture(s, FIG_DIR / "explainability_maps.png", 0.3, 1.1, 8.3, 5.6)
picture(s, FIG_DIR / "embedding_tsne.png", 8.7, 1.5, 4.5, 3.0)
if len(probe_tab) == 2:
    bullets(s, [f"Domain probe accuracy: ArcFace-kNN {probe_tab.domain_probe_balanced_acc.iloc[0]:.2f} vs DA-ArcNet {probe_tab.domain_probe_balanced_acc.iloc[1]:.2f} (0.5 = invariant)"],
            left=8.7, top=4.7, width=4.4, height=1.8, size=14)

s = new_slide("Confusion matrices and error analysis")
picture(s, FIG_DIR / "confusion_matrix_pooled.png", 0.3, 1.1, 6.6, 5.8)
picture(s, FIG_DIR / "error_analysis.png", 7.0, 1.3, 6.0, 3.0)
bullets(s, ["Threshold from held-out SOURCE normals only -> more false alarms on target domain"], left=7.0, top=4.6, width=6.0, height=1.8, size=15)

s = new_slide("Limitations and future work")
bullets(s, ["Single fixed threshold; only 10 target clips per section; ToyADMOS2 limited to ToyCar / ToyTrain",
            "Baselines are re-implementations, not official challenge systems; few seeds",
            "Grad-CAM qualitative only",
            "Future: per-domain calibration, foundation-model distillation, outlier exposure, INT8 on edge device, DCASE 2023-25 settings"])

s = new_slide("Conclusion")
bullets(s, [F[0].replace("\\%", "%").replace("$", ""), F[3].replace("\\%", "%").replace("$", ""),
            "Code, notebook, paper and slides are in the project repository; the live demo runs on any uploaded wav clip."], size=19)

prs.save(str(OUT_DIR / "ASD_24I461_presentation.pptx"))
print("Saved", OUT_DIR / "ASD_24I461_presentation.pptx", "|", len(prs.slides), "slides")
dbg_end("presentation", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 25. Live demo: upload a clip, get a verdict and an explanation
# Select the machine and the section, upload a `.wav` (or use one of the dataset clips). The demo returns the anomaly scores of the baseline autoencoder, DA-ArcNet and the fusion with their thresholds, plus the residual map and Grad-CAM.

# %%
_dbg_tok = dbg_start("demo")          # --- debug (auto): banner + timer ---
def load_artifacts(machine):
    ae, prop, st, rf = load_machine_models(machine)
    info = json.load(open(MODEL_DIR / f"{machine}_info.json"))
    return ae, prop, st, rf, info


def predict_logmel(art, lm, section):
    ae, prop, st, rf, info = art
    F1 = apply_norm(lm[None], st)
    s_ae = float(ae_clip_scores(ae, F1, CFG)[0])
    s_pr = float(knn_scores(embed_clips(prop, F1, CFG), np.array([section]), rf["prop"], rf["sections"], CFG["KNN_K"])[0])
    s_fu = float(fuse(s_pr, s_ae, info["refs"][str(section)]))
    out = {}
    for k, s in [("ae", s_ae), ("prop", s_pr), ("full", s_fu)]:
        th = info["thresholds"][f"{section}|{k}"]
        out[k] = dict(score=s, threshold=th, verdict="ANOMALY" if s > th else "normal")
    return out, F1[0]


def explain_figure(art, lm, F1, section):
    ae, prop, st, rf, info = art
    res = ae_residual_map(ae, F1, CFG)
    cam = gradcam_clip(prop, F1, rf["prop"][rf["sections"] == section])
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    axs[0].imshow(lm.T, origin="lower", aspect="auto", cmap="magma"); axs[0].set_title("log-mel input")
    axs[1].imshow(res.T, origin="lower", aspect="auto", cmap="inferno"); axs[1].set_title("autoencoder residual")
    axs[2].imshow(lm.T, origin="lower", aspect="auto", cmap="gray"); axs[2].imshow(cam.T, origin="lower", aspect="auto", cmap="jet", alpha=0.45)
    axs[2].set_title("DA-ArcNet Grad-CAM")
    for a in axs:
        a.set_xlabel("frame"); a.set_ylabel("mel bin")
    plt.tight_layout()
    return fig


def demo_predict(machine, wav_path, section):
    art = load_artifacts(machine)
    lm = logmel(read_wave(wav_path))
    out, F1 = predict_logmel(art, lm, int(section))
    txt = "\n".join(f"{MODEL_NAMES[k]:<26} score {v['score']:.4g} | threshold {v['threshold']:.4g} -> {v['verdict']}" for k, v in out.items())
    return txt, explain_figure(art, lm, F1, int(section))


# --- offline demo on dataset clips (works without Gradio)
try:
    dm = MACHINES[0]
    meta_d, _ = get_data(dm)
    if not has_audio(dm):
        download_machine(dm)
        meta_d = parse_meta(dm)
    pick = pd.concat([meta_d[(meta_d.split == "test") & (meta_d.label == 0) & (meta_d.domain == "source") & (meta_d.section == 0)].head(1),
                      meta_d[(meta_d.split == "test") & (meta_d.label == 1) & (meta_d.domain == "source") & (meta_d.section == 0)].head(1),
                      meta_d[(meta_d.split == "test") & (meta_d.label == 1) & (meta_d.domain == "target") & (meta_d.section == 0)].head(1)])
    for _, row in pick.iterrows():
        wav = DATA_DIR / dm / row["name"] if (DATA_DIR / dm / row["name"]).exists() else next((DATA_DIR / dm).rglob(row["name"]))
        txt, fig = demo_predict(dm, wav, row.section)
        print(f"\n{row['name']}  [true: {'ANOMALY' if row.label else 'normal'}]\n{txt}")
        plt.show()
except Exception as e:
    print("Offline demo skipped:", e)

# --- interactive demo (Gradio)
try:
    import gradio as gr
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=False)
    try:
        import gradio as gr
    except Exception:
        gr = None
if gr is not None and IN_COLAB:
    ui = gr.Interface(fn=demo_predict,
                      inputs=[gr.Dropdown(MACHINES, value=MACHINES[0], label="Machine"), gr.Audio(type="filepath", label="Machine sound (.wav)"),
                              gr.Dropdown([0, 1, 2], value=0, label="Section")],
                      outputs=[gr.Textbox(label="Scores and verdict", lines=4), gr.Plot(label="Explanation")],
                      title="Anomalous sound detection - DA-ArcNet", description="Upload a 10-s machine sound; the system says normal / anomaly.")
    ui.launch(share=True, prevent_thread_lock=True)
else:
    print("Gradio demo is launched in Colab only; use demo_predict(machine, wav_path, section) elsewhere.")
dbg_end("demo", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 26. Summary and export

# %%
_dbg_tok = dbg_start("summary")          # --- debug (auto): banner + timer ---
print("=" * 78)
print("SUMMARY")
print("=" * 78)
o = over_tab.set_index("model").loc[MODELS]
for m in MODELS:
    r = o.loc[m]
    print(f"{MODEL_NAMES[m]:<28} AUC src {r.AUC_source:.3f} | AUC tgt {r.AUC_target:.3f} | pAUC {r.pAUC:.3f} | official {r.official:.3f}")
best = o.official.idxmax()
print(f"\nBest model overall (official score): {MODEL_NAMES[best]} ({o.official.max():.3f})")
print("\nBest model per machine (official score):")
for mc in MACHINES:
    g = mach_tab[mach_tab.machine == mc].set_index("model").official.loc[MODELS]
    print(f"  {mc:<9} {MODEL_NAMES[g.idxmax()]:<28} {g.max():.3f}")
print("\nMean AUC drop from source to target domain (domain-shift penalty):")
for m in MODELS:
    print(f"  {MODEL_NAMES[m]:<28} {o.AUC_source[m] - o.AUC_target[m]:+.3f}")
pooled = cls_tab[(cls_tab.machine == "ALL") & (cls_tab.domain == "all")].set_index("model")
print("\nThreshold-based (pooled, both domains):")
for m in MODELS:
    r = pooled.loc[m]
    print(f"  {MODEL_NAMES[m]:<28} Acc {r.Accuracy:.3f} | Prec {r.Precision:.3f} | Recall {r.Recall:.3f} | Spec {r.Specificity:.3f} | F1 {r.F1:.3f} | MCC {r.MCC:.3f}")
print("\nAuto-worded findings:")
print("\n".join(" * " + x.replace("\\%", "%").replace("$", "") for x in F))
print("\nExecution log of this session (seconds per cell):")
display(pd.DataFrame(CELL_LOG))
print("\nFiles saved in", OUT_DIR)
for sub in (FIG_DIR, TAB_DIR, MODEL_DIR, PAPER_DIR):
    print(f"  {sub.name}/: {len(list(sub.iterdir()))} items")
dbg_end("summary", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %%
_dbg_tok = dbg_start("export")          # --- debug (auto): banner + timer ---
zip_base = str(WORK / "ASD_results_24I461")
archive = shutil.make_archive(zip_base, "zip", OUT_DIR)
print("Created", archive, f"({os.path.getsize(archive) / 1e6:.1f} MB)")
try:
    from google.colab import files
    files.download(archive)
except Exception as e:
    print("Automatic download only works in Colab:", e)
dbg_end("export", _dbg_tok)          # --- debug (auto): elapsed time + variables created by this cell ---

# %% [markdown]
# ## 27. Conclusion
# The auto-worded findings above (Section 26) summarise what *this* run shows. When you write the conclusion in the report, state: (i) the official score and target-domain AUC of the proposed model against the best baseline **with the confidence interval and the p-value**, (ii) which ablation component helped and which did not, (iii) whether the cross-dataset transfer worked in each direction, (iv) the cost in parameters and milliseconds, and (v) the limitations listed in Section 22. If a component did not help, say so - an honest negative result in the ablation is acceptable; an unsupported claim is not.
# 
# **Dataset references:** MIMII DG (Dohi et al., 2022, zenodo 6529888), ToyADMOS2 (Harada et al., 2021, zenodo 4580270) and the DCASE 2022 Challenge Task 2 development dataset (zenodo 6355122).
