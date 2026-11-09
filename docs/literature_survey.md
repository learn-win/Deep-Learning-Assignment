# Literature survey (verified references, 2022-2026)

Every entry was checked against its arXiv abstract page during project preparation. Descriptions only restate what the title/abstract says.

| # | Authors | Year | Title | What it contributes | Limitation relevant to this work |
|---|---|---|---|---|---|
| 1 | Dohi et al. | 2022 | Description and Discussion on DCASE 2022 Challenge Task 2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring Applying Domain Generalization Techniques (arXiv:2206.05876) | Defines the domain-generalisation ASD benchmark (source/target domains, AUC and pAUC) and the AE and MobileNetV2 baselines used in this work. | Benchmark paper; baselines do not address domain shift explicitly. |
| 2 | Dohi et al. | 2022 | MIMII DG: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection for Domain Generalization Task (arXiv:2205.13879) | MIMII DG dataset: five machine types, three sections per type, each section a different kind of domain shift. | Dataset paper. |
| 3 | Liu et al. | 2022 | Anomalous Sound Detection using Spectral-Temporal Information Fusion (arXiv:2201.05510) | STgram-MFN: fuses spectrogram and temporal features for self-supervised machine-ID classification. | Uses only the in-domain machine ID task; no explicit domain-invariance. |
| 4 | Zeng et al. | 2023 | Joint Generative-Contrastive Representation Learning for Anomalous Sound Detection (arXiv:2305.12111) | GeCo: self-attention predictive autoencoder plus multi-task contrastive objective; evaluated on DCASE 2020 Task 2. | Evaluated on a single benchmark year without domain shift. |
| 5 | Zhang et al. | 2023 | Anomalous Sound Detection Using Self-Attention-Based Frequency Pattern Analysis of Machine Sounds (arXiv:2308.14063) | Self-attention over frequency patterns, combined with spectral and temporal information; DCASE 2020 Task 2. | No domain-shift setting evaluated. |
| 6 | Dohi et al. | 2023 | Description and Discussion on DCASE 2023 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring (arXiv:2305.07828) | Extends the benchmark to the first-shot setting. | Benchmark paper. |
| 7 | Wilkinghoff | 2023 | Self-Supervised Learning for Anomalous Sound Detection (arXiv:2312.09578) | Proposes feature exchange (FeatEx) as a self-supervised approach and reports state-of-the-art results on DCASE 2023. | Ensembles of auxiliary-task embeddings; no cross-dataset transfer study. |
| 8 | Wilkinghoff | 2024 | AdaProj: Adaptively Scaled Angular Margin Subspace Projections for Anomalous Sound Detection with Auxiliary Classification Tasks (arXiv:2403.14179) | AdaProj loss: projects embeddings onto class-specific subspaces with an angular margin; evaluated on DCASE 2022 and 2023. | Embedding-loss improvement; domain shift handled only through data and scoring. |
| 9 | Nishida et al. | 2024 | Description and Discussion on DCASE 2024 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring (arXiv:2406.07250) | DCASE 2024 first-shot task description. | Benchmark paper. |
| 10 | Jiang et al. | 2024 | AnoPatch: Towards Better Consistency in Machine Anomalous Sound Detection (arXiv:2406.11364) | Addresses consistency between pre-training and ASD fine-tuning (as stated in the title). | Relies on large pre-trained models. |
| 11 | Saengthong and Shinozaki | 2024 | Deep Generic Representations for Domain-Generalized Anomalous Sound Detection (arXiv:2409.05035) | GenRep: generic features from a large pre-trained extractor with kNN scoring, MemMixup and domain normalisation; Official Score 73.79% on the DCASE 2023 Task 2 evaluation set. | Large pre-trained extractor; heavy for edge deployment. |
| 12 | Zheng et al. | 2024 | Improving Anomalous Sound Detection via Low-Rank Adaptation Fine-Tuning of Pre-Trained Audio Models (arXiv:2409.07016) | LoRA fine-tuning of pre-trained audio models for ASD. | Large pre-trained models; high compute. |
| 13 | Purohit et al. | 2024 | MIMII-Gen: Generative Modeling Approach for Simulated Evaluation of Anomalous Sound Detection System (arXiv:2409.18542) | Generative simulation of data for evaluating ASD systems (as stated in the title). | Evaluation methodology rather than detection model. |
| 14 | Meng et al. | 2024 | Contrastive Learning with Spectrum Information Augmentation in Abnormal Sound Detection (arXiv:2509.15570) | Contrastive learning with spectrum-level augmentation for abnormal sound detection. | Augmentation-based; no explicit domain-adversarial component. |
| 15 | Wilkinghoff et al. | 2025 | Handling Domain Shifts for Anomalous Sound Detection: A Review of DCASE-Related Work (arXiv:2503.10435) | Reviews domain generalisation work for ASD from the DCASE challenges; domain shifts such as microphone type or placement can affect audio more than anomalies do. | Review; motivates robust, domain-aware design. |
| 16 | Nishida et al. | 2025 | Description and Discussion on DCASE 2025 Challenge Task 2: First-shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring (arXiv:2506.10097) | DCASE 2025 first-shot task description. | Benchmark paper. |
| 17 | Han et al. | 2025 | Exploring Self-Supervised Audio Models for Generalized Anomalous Sound Detection (arXiv:2508.12230) | Fine-tunes speech/audio self-supervised models with LoRA, a Machine-aware Group Adapter and a dual-level contrastive loss; evaluated on DCASE 2020-2024 datasets. | Large pre-trained backbones; no lightweight design. |
| 18 | Gao et al. | 2026 | UD-ASD: A Unified Diffusion Model for Anomalous Sound Detection (arXiv:2607.12576) | Single diffusion model on log-mel spectrograms with machine-ID condition embeddings and GMM scoring; improves over the baseline by 3.44% AUC and 2.52% pAUC on DCASE 2022 Task 2. | Generative model; domain shift not handled explicitly. |


## Dataset and method references

* Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, Shoichiro Saito (2021). ToyADMOS2: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions. arXiv preprint arXiv:2106.02369.
* Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Fran\ccois Laviolette, Mario Marchand, Victor Lempitsky (2016). Domain-Adversarial Training of Neural Networks. Journal of Machine Learning Research, vol. 17, no. 59, pp. 1--35.
* Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. Proc. IEEE/CVF CVPR, pp. 4690--4699.
* Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. Proc. IEEE ICCV, pp. 618--626.
* Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. Proc. IEEE/CVF CVPR, pp. 4510--4520.
* Diederik P. Kingma, Jimmy Ba (2015). Adam: A Method for Stochastic Optimization. Proc. ICLR (arXiv:1412.6980).


## Research gap
1. **Heavy models.** Recent high-scoring systems rely on large pre-trained backbones, LoRA fine-tuning or ensembles [11, 12, 17]; a *lightweight* model with a strong domain-shift mechanism is less explored.
2. **Implicit domain handling.** Works improve the embedding loss [8] or use memory banks and normalisation [11], but do not train the embedding to be *explicitly invariant* to the source/target domain while keeping an angular-margin condition classifier.
3. **Single-benchmark evaluation.** Cross-dataset transfer (MIMII DG <-> ToyADMOS2) and extra-noise robustness are rarely reported alongside the main score.
4. **Limited analysis.** Abstracts focus on AUC / pAUC; explainability and error-case analysis get little attention.

## Objectives
* **O1.** Design a lightweight ASD model explicitly robust to domain shift (DA-ArcNet).
* **O2.** Compare with three baselines on MIMII DG and ToyADMOS2 with the official AUC / pAUC protocol.
* **O3.** Test cross-dataset transfer in both directions.
* **O4.** Quantify each component by ablation and robustness by additive-noise tests.
* **O5.** Provide statistical evidence (bootstrap CIs, multiple seeds, Wilcoxon) and explainability (Grad-CAM, AE residual maps, t-SNE).
