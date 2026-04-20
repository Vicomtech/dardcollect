# Model Card — OpenAI Whisper Small (`openai_whisper_small.pt`)

Technical documentation structured in accordance with EU AI Act Annex IV.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** Automatic speech recognition (ASR) — transcribing spoken audio to text in 98+ languages, with optional speech translation to English.  
**Provider:** OpenAI (https://openai.com).  
**Version:** Whisper Small (244 M parameters), multilingual variant.  
**Usage in this pipeline:** Optional post-processing step (`scripts/transcribe_clips.py`) — disabled by default (`enable_transcription: false`). Not called during clip extraction or face-crop extraction.

### 1b. Interaction with Hardware & Software
- Runtime: PyTorch (via the `openai-whisper` Python package) or `faster-whisper` (CTranslate2 backend).
- Upstream: receives audio extracted from video clips (typically 2–60 second segments).
- Downstream: transcription text stored alongside clip metadata.
- No external network access at inference time.

### 1c. Software Versions
- PyTorch checkpoint format (`.pt`), compatible with `openai-whisper ≥ 20230918`.
- Audio pre-processing: 16 kHz mono, log-Mel spectrogram (80 bins, 25 ms window, 10 ms hop).

### 1d. Distribution Form
PyTorch checkpoint downloaded automatically by the `openai-whisper` library from:  
https://openaipublic.azureedge.net/main/whisper/models/  
Also available on Hugging Face: https://huggingface.co/openai/whisper-small  
Included here as a cached local copy for offline operation.

### 1e. Hardware Requirements
- CPU inference: feasible but slow (roughly 1× real-time).
- Recommended: NVIDIA GPU with CUDA 12.x, ≥ 2 GB VRAM (Whisper Small is compact).
- Model file size: 483 MB.

### 1g. Interface for Deployers
Called via `scripts/transcribe_clips.py`, which is a standalone post-processing script. Not integrated into the real-time extraction pipeline. Produces `.txt` or `.json` transcription files alongside clip videos.

### 1h. Usage Notes
- Maximum native audio segment length: 30 seconds (longer audio is chunked).
- Language is auto-detected or can be forced via `--language` flag.
- Temperature scheduling and beam search mitigate but do not eliminate repetitive output.
- Not suitable for real-time transcription in this deployment configuration.

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
- Trained by OpenAI using large-scale **weak supervision**: audio is paired with transcripts harvested from the internet without manual quality checks.
- Architecture is a standard Transformer encoder-decoder.
- Multi-task training: ASR + speech translation + language identification in a single model.

### 2b. Design Specifications & Algorithm
- **Encoder:** Audio is converted to log-Mel spectrogram → two convolutional stem layers → Transformer encoder (8 layers, 512 d_model, 8 heads).
- **Decoder:** Transformer decoder (8 layers) auto-regressively generates BPE tokens.
- **Special tokens** guide task switching: `<|transcribe|>`, `<|translate|>`, `<|notimestamps|>`, language tokens.
- **Decoding:** Beam search (default beam=5), temperature fallback, timestamp prediction.
- Total parameters: 244 M.

### 2c. System Architecture & Compute
- File: 483 MB PyTorch checkpoint (float32 weights).
- Inference compute: ~0.5 GFLOPs per second of audio on GPU.
- Memory footprint: ~1 GB GPU VRAM at float32; ~500 MB at float16.

### 2d. Training Data
| Source | Volume | Description |
|--------|--------|-------------|
| Internet audio + transcripts | 680,000 hours | Multilingual, scraped from the web |
| English ASR | 438,000 h (65%) | English audio with English transcripts |
| Speech translation | 126,000 h (18%) | Non-English audio with English transcripts |
| Non-English ASR | 117,000 h (17%) | Non-English audio with native language transcripts |

**Languages covered:** 98+ languages. Representation is heavily skewed toward English and high-resource European languages.  
**Labelling:** Weak supervision — transcripts obtained automatically from web sources, not manually verified.  
**Known data biases:** Low-resource languages and non-Western dialects are substantially under-represented. Training data provenance is not fully disclosed by OpenAI.

### 2e. Human Oversight
Within this pipeline, transcription is a supplementary metadata tool, not used for any automated filtering or selection decision. Outputs are reviewed by the end user. No consequential decision is derived solely from transcription.

### 2g. Validation & Testing
Official benchmark results (Whisper Small multilingual):

| Benchmark | Metric | Value |
|-----------|--------|-------|
| LibriSpeech test-clean | WER | 3.43% |
| LibriSpeech test-other | WER | 7.63% |
| Common Voice 11.0 | WER | 87.3% |

Note: Common Voice WER is high because it includes many low-resource languages where the model performs poorly.  
Performance on silent-era film audio (scratchy, mono, non-speech artifacts, early recording equipment) has not been benchmarked and is expected to be significantly worse.

### 2h. Cybersecurity
- Model loaded read-only at runtime.
- Checkpoint downloaded from OpenAI's official CDN or Hugging Face; no integrity verification is performed at load time.
- Audio input is processed locally; no data is sent to external services.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- High-accuracy transcription for clear English and major European language speech.
- Automatic language detection across 98 languages.
- Timestamp-level word alignment (with `--word_timestamps`).
- Noise robust relative to earlier ASR models due to large-scale weak supervision.

### Limitations
- **Hallucination:** The model may generate plausible-sounding text not present in the audio. This is an inherent property of the seq2seq architecture trained with weak supervision. Risk is higher on low-quality or non-speech audio.
- **Low-resource language performance:** Substantially degraded for languages with less than ~1,000 hours in the training set.
- **Demographic disparities:** Higher WER observed for non-native accents, older speakers, children, and non-standard dialects. No formal audit has been published.
- **Historical film audio:** Silent films have no dialogue audio; early sound films (1927–1955 era covered by this project) often have poor acoustic quality, background music, or non-standard microphone placement. Performance is expected to be well below LibriSpeech benchmarks.
- **Repetition:** Sequence-to-sequence models are prone to looping. Partially mitigated by beam search but not eliminated.
- **No speaker diarisation:** The model produces a single transcript stream; it does not separate multiple speakers.
- **Not real-time:** Batch inference only in this deployment.

### Foreseeable Unintended Outcomes
- Hallucinated text that could be mistaken for actual speech content in archive metadata.
- Inaccurate transcription of proper nouns, historical terminology, or non-standard pronunciation common in early cinema.
- Language mis-identification for multilingual or code-switched audio segments.

### Input Data Specifications
- Input: audio at any sample rate (resampled internally to 16 kHz mono).
- Maximum segment: 30 seconds natively; longer segments chunked with overlap.
- Minimum useful duration: ~1 second.

---

## 4. Performance Metrics Rationale
Word Error Rate (WER) is the standard ASR metric, measuring edit distance between hypothesis and reference transcription normalised by reference length. It is appropriate for this use case where transcription accuracy directly affects metadata quality. WER is reported on standardised benchmarks (LibriSpeech) to enable comparison; project-specific performance on historical film audio should be evaluated separately.

---

## 5. Risk Management
Within this pipeline the model is used for supplementary metadata enrichment on archival video. No automated decision with legal or significant personal effect is made from transcriptions. Risks are mitigated by:
- Transcription is disabled by default (`enable_transcription: false`).
- Output is advisory text; the pipeline does not gate clip selection on transcription content.
- Outputs should be manually reviewed before use in any publication or archival record.

---

## 6. Known Lifecycle Changes
`openai_whisper_small.pt` corresponds to the September 2023 release of the `openai-whisper` package. OpenAI has released larger and updated variants (medium, large-v2, large-v3). If transcription quality is critical, consider upgrading to `large-v3` at the cost of higher compute. No fine-tuning on this project's data has been performed.

---

## 7. Standards & Specifications Applied
- LibriSpeech evaluation protocol (https://www.openslr.org/12).
- Common Voice evaluation (https://commonvoice.mozilla.org).
- NIST SCTK scoring toolkit conventions.

---

## Citation
```bibtex
@article{radford2022whisper,
  title   = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author  = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg
             and McLeavey, Christine and Sutskever, Ilya},
  journal = {arXiv preprint arXiv:2212.04356},
  year    = {2022},
  doi     = {10.48550/ARXIV.2212.04356}
}
```
Paper: https://arxiv.org/abs/2212.04356
