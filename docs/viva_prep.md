# Viva preparation - DA-ArcNet (Author: Ardhanarieswarar 24I461; team: Gokul Kannan 24I463, Jafer Sathik B 24I464, Madhan M 24I466; PSG College of Technology, Department of Information Technology)

Answer in your own words and with **your** numbers (tables in `ASD_project/tables/`). Short model answers below.

## Problem and data
**Why train on normal sounds only?** Faults are rare and take many forms; labelled anomalies for every fault cannot be collected. We learn what normal sounds like and flag deviations.

**What is domain shift here?** Operating speed/load, microphone position or background noise differ between the *source* (many training clips) and *target* (only 10 training clips) domain. These changes can alter the sound more than a fault does.

**Which datasets?** MIMII DG (fan, gearbox, bearing, slider, valve) and ToyADMOS2 (ToyCar, ToyTrain from the DCASE 2022 development set, because the raw 169 GB record needs a separate mixer). Each machine has 3 sections; per section 990 source + 10 target normal training clips; test 50 normal + 50 anomalous per domain.

**Is there leakage?** Normalisation is fitted on training clips only; 10 % of source normal clips are held out for threshold, early stopping and score scaling; held-out clips are not kNN references; the code asserts that train/held-out/test indices are disjoint.

## Method
**What are the log-mel features?** FFT 1024, hop 512, 64 mel bands, in dB (16 kHz audio).

**What is AdaCMN?** `x - sigmoid(a_f) * mean_t(x)`: a learnable per-band blend between keeping the level and removing the clip's mean spectrum, to cope with constant level/noise offsets. It is an own design; the ablation tests whether it helps.

**What is ArcFace and why use it?** A softmax with an additive angular margin on cosine similarities; it makes embeddings of the same class compact and different classes far apart on a hypersphere, which suits kNN scoring. Classes = section x operating-condition attributes (from file names).

**What does the gradient-reversal layer do?** Identity forward, gradient times -lambda backward. The domain classifier learns to tell source from target, the encoder is pushed the opposite way, so the embedding carries little domain information (checked with the domain probe).

**Why oversample target clips?** Only 10 per section; without oversampling the adversary would rarely see the target domain.

**How is the anomaly score computed?** Cosine kNN (k=3) between a test clip's crop embeddings and normal reference embeddings of the same section; average over crops.

**How is the threshold chosen?** Gamma distribution fitted to held-out normal source scores; 90th percentile. No anomalous or test data used.

**What is the fusion?** z-score of the log scores of DA-ArcNet and the autoencoder (statistics from held-out normals), averaged, exponentiated.

## Evaluation
**AUC vs pAUC?** AUC over all thresholds; pAUC only for FPR <= 0.1 (low false-alarm region, important in factories). Official score = harmonic mean of AUC_source, AUC_target and pAUC over sections.

**Baselines?** DCASE 2022 AE, DCASE 2022 MobileNetV2, ArcFace-kNN. All re-implemented with the same budget, so not identical to the authors' systems - say this openly.

**How do you know a difference is real?** Bootstrap 95 % CIs, several seeds on two machines, paired Wilcoxon tests (descriptive: machine-sections are not an i.i.d. sample). If intervals overlap, call the result indicative.

**What did the ablation show?** (Use your table.) State which of AdaCMN / adversary / fusion helped, and where they did not.

**What does cross-dataset mean here?** The embedding network is trained on one dataset's normal clips and never on the test dataset; only normal training clips of the test machine build the kNN bank.

**How do you explain the model?** AE residual maps (time-frequency reconstruction error), Grad-CAM on the anomaly score, t-SNE and a linear domain probe on the embedding. They show where the model responds, not verified fault locations.

## Limitations and honesty
* Threshold from source data only -> false alarms on target normals.
* Only 10 target clips per section; few seeds; re-implemented baselines.
* ToyADMOS2 = ToyCar/ToyTrain only.
* If the proposed model does not beat a baseline somewhere, say so and explain with the per-section table and error analysis.

## Quick facts to remember
Parameters and ms/clip: see `tables/complexity_summary.csv`. TensorFlow saves `.keras`/SavedModel (no `.pt`). Run `QUICK_TEST` before the full run.
