# Model Card — Expression Neutrality Models

Technical documentation structured in accordance with EU AI Act Annex IV.

This card covers three model files in `persondet/models/`:
- `enet_b0_8_best_vgaf_embed_zeroed.onnx` — EfficientNet-B0 emotion embedder (HSEmotion)
- `enet_b2_8_embed_zeroed.onnx` — EfficientNet-B2 emotion embedder (HSEmotion)
- `hse_1_2_C_adaboost.yml.gz` — AdaBoost classifier on HSEmotion embeddings (Grimmer/ITWM)

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** **Facial expression neutrality assessment** — classifying whether a face image shows a neutral expression or an emotional expression (happiness, anger, surprise, etc.). Used to penalise non-neutral expressions in biometric face images.  
**HSEmotion authors:** Andrey Savchenko — National Research University Higher School of Economics (HSE).  
**AdaBoost classifier:** BSI/Fraunhofer ITWM (OFIQ project), trained on top of HSEmotion embeddings.  
**ONNX packaging:** BSI/Fraunhofer ITWM as part of the OFIQ reference implementation.

### 1b. Interaction with Hardware & Software
- Runtime: ONNX Runtime ≥ 1.16.0 (EfficientNet models) + OpenCV `ml` module (AdaBoost `.yml.gz`).
- Intended downstream use (OFIQ): `ExpressionNeutrality` quality measure — produces a score in [0, 100] where 100 = perfectly neutral expression.
- Pipeline: image → EfficientNet-B0 embedding → EfficientNet-B2 embedding → concatenated → AdaBoost classifier → neutrality score.
- No external network access at inference time.

### 1c. Software Versions
- `enet_b0_8_best_vgaf_embed_zeroed.onnx`: EfficientNet-B0, 8-class emotion embedder, `_zeroed` = emotion-class logits zeroed to retain only the embedding part.
- `enet_b2_8_embed_zeroed.onnx`: EfficientNet-B2, 8-class emotion embedder, same convention.
- `hse_1_2_C_adaboost.yml.gz`: OpenCV AdaBoost binary classifier trained on HSEmotion B0+B2 features.

### 1d. Distribution Form
HSEmotion models: originally published at https://github.com/HSE-asavchenko/face-emotion-recognition  
AdaBoost classifier + ONNX packaging: OFIQ project (https://github.com/BSI-OFIQ/OFIQ-Project)

### 1e. Hardware Requirements
- EfficientNet-B0: ~20 MB. EfficientNet-B2: ~35 MB. AdaBoost: < 1 MB.
- Recommended: GPU for EfficientNet forward passes; AdaBoost runs on CPU.

### 1g. Interface for Deployers
Used by `scripts/annotate_face_quality.py` (`_expression_neutrality_score`) to compute the OFIQ `ExpressionNeutrality` quality measure.

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
- Architecture: **EfficientNet-B0/B2** — compound-scaled CNN balancing depth, width, and resolution. Fine-tuned for facial expression recognition.
- **HSEmotion pipeline:** Two EfficientNet backbones (B0, B2) trained separately on emotion datasets, then used as feature extractors. Their embeddings are concatenated and fed to a downstream classifier.
- **AdaBoost:** Boosted ensemble of weak classifiers trained to distinguish neutral from non-neutral expression on the combined HSEmotion features.
- Training: PyTorch (EfficientNet) + OpenCV `ml` (AdaBoost).

### 2b. Design Specifications & Algorithm
- EfficientNet-B0 input: `[batch, 3, H_B0, W_B0]` float32 (exact size per HSEmotion convention, ~224×224).
- EfficientNet-B2 input: `[batch, 3, H_B2, W_B2]` float32.
- `_zeroed` variant: final classification logits replaced with zeros; only the penultimate embedding is used.
- AdaBoost: binary classification (neutral / non-neutral) on concatenated B0+B2 embeddings.
- OFIQ sigmoid calibration maps AdaBoost score to [0, 100] (see `ofiq_config.jaxn`).

### 2d. Training Data
**HSEmotion (EfficientNet):**

| Dataset | Description |
|---------|-------------|
| AffectNet | ~450,000 images, 8 emotion classes (neutral, happy, sad, surprise, fear, disgust, anger, contempt), web-scraped |
| VGGFace2-FER | FER labels applied to VGGFace2 identities |
| AFEW-VA | Video-based arousal/valence annotations |

**AdaBoost (OFIQ):** Trained on a face dataset with expert neutrality annotations; details not publicly disclosed by BSI/ITWM.

AffectNet has known annotation noise (~10–15% label error rate) and demographic skews in the crowdsourced emotion labels.

### 2e. Human Oversight
Wired into `scripts/annotate_face_quality.py`; neutrality scores are written to the `.quality.json` sidecar for human review.

### 2g. Validation & Testing
HSEmotion EfficientNet-B0 on AffectNet-8 val: ~63% accuracy (8-class emotion classification).  
Neutrality detection accuracy on OFIQ test set: not publicly disclosed.

### 2h. Cybersecurity
EfficientNet weights loaded via ONNX Runtime (sandboxed). AdaBoost loaded via OpenCV `ml`. Both read-only.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Two-model ensemble (B0 + B2) provides more robust emotion embeddings than a single model.
- Efficient inference: EfficientNet-B0/B2 are compact models relative to their accuracy.

### Limitations
- **Label noise:** AffectNet emotion labels are crowd-sourced and noisy; neutrality boundary is subjective.
- **Cultural variation:** Expressions of neutrality vary across cultures; model trained predominantly on Western web images.
- **Historical footage:** Silent-era acting conventions often involve exaggerated expressions; model not trained on this domain.
- **AdaBoost:** Requires OpenCV `ml` module, separate from the ONNX Runtime infrastructure used elsewhere.

### Foreseeable Unintended Outcomes
- Subtle smiles common in portrait photography may be classified as non-neutral, reducing scores for otherwise high-quality images.
- Exaggerated "neutral" acting faces in early cinema may be misclassified.

---

## 7. Standards & Specifications Applied
- ISO/IEC 29794-5 (OFIQ): `ExpressionNeutrality` measure.
- HSEmotion: Savchenko, "HSEmotion: Efficient Facial Sentiment Analysis Framework", ICMR 2022.
- EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
- OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project

---

## Citation
```bibtex
@inproceedings{savchenko2022hsemotion,
  title     = {HSEmotion: Efficient Facial Sentiment Analysis Framework},
  author    = {Savchenko, Andrey V.},
  booktitle = {Proceedings of the ACM International Conference on Multimedia Retrieval (ICMR)},
  year      = {2022}
}

@inproceedings{tan2019efficientnet,
  title     = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author    = {Tan, Mingxing and Le, Quoc V.},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2019}
}
```
HSEmotion repository: https://github.com/HSE-asavchenko/face-emotion-recognition  
OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project
