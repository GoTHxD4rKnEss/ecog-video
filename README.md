# 🧠 ECoG–Video Analysis Project  
**BR41N.IO Hackathon 2025 – Team Neural Vision**

---

## 🎯 Project Overview
We built a complete **ECoG (Electrocorticography) video-analysis pipeline** that processes high-density brain recordings synchronized with movie stimuli, extracts meaningful features, and visualizes real-time cortical activity.  

Our system integrates **signal processing, feature extraction, machine learning, and interactive visualization** into a unified, reproducible workflow — built collaboratively during the BR41N.IO Hackathon.

---

## 📊 1. Data Overview & Exploration
- **Dataset**: 160-channel ECoG data (1200 Hz) synchronized with 30 FPS video (252 s)
- **Coverage**: Motor, sensory, and association cortices
- **Quality**: High-SNR signals, 160 active channels
- **Timeline**: 268.4 s total recording with **252 stimulus trials**
- **Key Finding**: 4 distinct stimulus periods with stable temporal structure

---

## 🔧 2. Advanced Preprocessing Pipeline
Our preprocessing stage transforms raw intracranial data into clean, analysis-ready signals.

**Signal Processing**
- Band-pass filter **0.5–150 Hz**
- Notch filter at **50/60 Hz**
- Common Average Reference (CAR) for global noise removal  
- Independent Component Analysis (ICA) for artifact rejection  

**Quality Control**
- Artifact detection using variance, amplitude, correlation, and spectral heuristics  
- Channels flagged as *good* or *bad* (`good_channels.npy`, `bad_channels.npy`)

**Trial Segmentation**
- 252 trials extracted: **–300 ms to +400 ms** around each stimulus onset  
- Baseline normalization (z-score / percent change / dB)  

**Results**
- 8 preprocessing runs with quality > 85 %

---

## ⚡ 3. Multi-Modal Feature Extraction
We computed several complementary feature types capturing neural information at multiple scales.

| Method | Description | Purpose |
|--------|--------------|----------|
| **A. Broadband Gamma (110–140 Hz)** | Hilbert envelope power | Primary neural activation metric |
| **B. Canonical Bands** | Theta, alpha, beta, gamma | Comparison across frequency bands |
| **C. CSP + LDA** | Spatial filtering & linear discrimination | Real-time decoding baseline |
| **D. Template Correlation** | Match to learned stimulus templates | Robust low-data decoding |
| **E. EEGNet CNN** | Compact deep neural network | End-to-end classification |
| **F. Transformer** | Temporal attention model | State-of-the-art decoding |

**Output:**  
Multiple feature representations optimized for different neural signatures.

---

## 🤖 4. Advanced Multiclass Machine Learning
Using manually verified video annotations:

- **7-class setup:** digit, kanji, face, body, object, hiragana, line  
- **14-class extension:** adds grayscale vs color variants  
- **Models:** Random Forest, SVM, Logistic Regression, Neural Nets, Ensemble  
- **Performance:**
  - 7-class → **89.6 % accuracy**, 100 % ROC-AUC  
  - 14-class → **85.2 % accuracy** (color discrimination)
- **Validation:** Cross-validation with stratified splits and balanced sampling  

---

## 🎬 5. Real-Time Video Annotation System
We developed a system that synchronizes decoded brain activity with the actual movie frames, generating **annotated videos** showing moment-to-moment neural engagement.

**Implemented Approaches**
1. Spatial Motor-Cortex Activation Mapping  
2. Gait-Phase Neural Signatures  
3. ERSP ( Event-Related Spectral Perturbation ) Overlays  
4. Enhanced Multi-Region Connectome Visualization  
5. Object Detection + Neural Correlation  
6. Brain-Atlas Mapping (via *nilearn*)  
7. Anatomical Region Activation Overlays  

**Deliverables**
- > 50 annotated videos linking visual scenes to neural dynamics  
- First system to merge **ECoG + computer-vision annotation in real time**

---

## 🌐 6. Interactive Web Application
A lightweight web dashboard provides exploration, visualization, and live updates.

**Hosted locally:** http://localhost:5001  

**Sections**
1. **Data Overview** – Electrode map & recording summary  
2. **Preprocessing Monitor** – Filter stages, QC metrics  
3. **Feature Extraction** – Multiple frequency-band visualizations  
4. **Modeling Results** – Confusion matrices, ROC curves  
5. **Video Annotations** – Real-time playback of synchronized data  
6. **Visualization Gallery** – Aggregated results and plots  

**UX Highlights**
- Interactive charts and toggles  
- Real-time updates  
- Clean UI for demo presentations  

---

## 📈 7. Results & Deliverables
| Category | Deliverable |
|-----------|-------------|
| **Experiments** | > 50 full runs across pipeline stages |
| **Figures & Charts** | > 200 publication-quality visualizations |
| **Videos** | > 50 annotated brain–video recordings |
| **Documentation** | Technical and user guides |
| **Code Quality** | Modular, tested, reproducible |
| **Repository** | All results organized under `/data/derivatives/` |
| **Drive Access** | Full dataset + outputs shared with team |

---

## 🎯 Key Achievements
- 🚀 **First complete ECoG–video synchronization system**  
- ⚡ **High decoding performance** (up to 89.6 % accuracy)  
- 🧩 **Real-time annotation** of cortical activity during video playback  
- 💡 **Multi-modal integration**: signal processing + ML + computer vision  
- 🧱 **Production-ready web app** with professional design  
- 🔁 **Reproducible pipeline** with automated outputs  

---

## 📱 Presentation Strategy
**Primary Demo Tool:** the local web dashboard  
(http://localhost:5001)

**Flow**
1. **Data Overview:** show electrode layout & quality  
2. **Preprocessing:** demonstrate filters and artifact removal  
3. **Feature Extraction:** highlight broadband gamma extraction  
4. **Modeling:** present multiclass decoding accuracy  
5. **Video Annotation:** play brain-activity overlays  

---

## 🧠 Scientific Motivation
High-gamma (70–150 Hz) power correlates strongly with **local cortical firing**, reflecting how populations of neurons respond to sensory input.  
By synchronizing these signals with a video stream, we can **decode perception and attention in real time** — turning raw brain data into interpretable, visualizable information.

---

## 🧪 Team Roles

| Role | Member | Task |
|------|---------|------|
| **Preprocessing** | Member A | Filtering, artifact rejection |
| **Synchronization** | Member B | Align ECoG ↔ video signals |
| **Feature Extraction** | **Aryan Babaie** | High-gamma envelope, visualization |
| **Modeling & ML** | Members C–D | Classification, evaluation |
| **Presentation & Docs** | All | Slides, reports, app demo |

---

## 🚀 Next Steps
1. Re-extract per-trial HG features using corrected offset  
2. Train ML models and evaluate decoding performance  
3. Integrate real-time annotation into web app  
4. Prepare final slide deck and demo videos  

---

## 🪪 Reference
> Kapeller et al. (2018). *Real-Time Detection and Discrimination of Visual Stimuli from Human Intracranial EEG.*  

Developed for the **IEEE SMC BR41N.IO Hackathon 2025**  
Team Neural Vision  

> *“From brain waves to perception — decoding the mind in real time.”*
