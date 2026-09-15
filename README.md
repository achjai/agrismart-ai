# AgriSmart AI — Intelligent Farm Advisory & Monitoring System

AgriSmart AI is an end-to-end decision support prototype built to assist smallholder farmers. It integrates remote-sensing telemetry, machine-learning classifiers, and automated generative AI advisory services into a single, farmer-friendly dashboard.

This repository fulfills the requirements of the **SIH 2026 Internal Hackathon (Problem Statement 1)**.

---

## 1. Modules Built

### Mandatory Core Task
- **[Core] Crop Disease Detection:** A fine-tuned ResNet50 computer vision model that classifies uploaded leaf images across 38 crop-disease categories (including healthy states). It provides immediate, actionable precautionary advice based on the detected condition.

### Optional Bonus Modules
- **[Bonus A] Crop Recommendation:** A Random Forest model that recommends ideal crops based on soil NPK levels, rainfall, and live environmental telemetry.
- **[Bonus B] Smart Irrigation:** Automated background logic that delays or schedules irrigation by correlating real-time soil moisture dropping below critical thresholds with local precipitation forecasts.
- **[Bonus C] Weather-Based Intelligence:** Live integration with the Open-Meteo API to pull current meteorological conditions and precipitation forecasts.
- **[Bonus D] Sustainability Score:** A live-calculated resource-efficiency metric (0-100) derived dynamically from how well current soil moisture, pH, and temperature align with optimal bands.
- **[Bonus E] Farmer Assistant (GenAI):** A conversational, plain-language assistant powered by a large language model (via OpenRouter). The model is "context-aware" — it receives the live sensor telemetry behind the scenes so it can give grounded, specific advice.
- **[Bonus F] IoT Integration (Simulated):** A background polling loop simulates live sensor feeds (soil moisture, temperature, humidity, pH), establishing the framework for physical hardware integration.

---

## 2. Setup & Run Instructions (Local Evaluation)

A judge can easily reproduce this environment and run the full stack locally in under 5 minutes.

### Prerequisites
- Python 3.10 or 3.11
- Git

### Step-by-Step Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/achjai/agrismart-ai.git
   cd agrismart-ai
   ```

2. **Run the Application:**

   - **Option A — 1-Click for Windows (Recommended):**
     Double-click the `run_windows.bat` file. It will automatically create a virtual environment, install all dependencies, download the model weights (~214MB) from Google Drive, and start the server. No manual steps needed.

   - **Option B — Manual Terminal:**
     ```bash
     pip install -r requirements.txt
     ```
     Then download the model weights manually:
     - Download from: [Google Drive Link](https://drive.google.com/file/d/1Hw-9exEnsLYtFLqYeOKogVtH6_XATw-8/view?usp=sharing)
     - Place the file at: `model/weights/RESNET50_FINETUNED.weights.h5`

     Then start the server:
     ```bash
     uvicorn src.main:app --host 127.0.0.1 --port 8000
     ```

3. **Test the UI:**
   Open your browser and navigate to `http://127.0.0.1:8000`. The frontend is served directly by the FastAPI backend.

> [!NOTE]
> The `.env` file with a working OpenRouter API key is already included in the repository. The GenAI assistant will work out of the box — no additional configuration needed.

---

## 3. Dataset & Source

- **Source Dataset:** New Plant Diseases Dataset (Augmented), derived from PlantVillage.
- **Source Link:** [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **License:** Copyright-authors (Open access for research/academic use).
- **Scale:** 70,295 training images and 17,572 validation images spanning 38 classes.
- **Split:** We utilized the pre-existing train/validation folder splits provided in the dataset.

---

## 4. Reported Metrics

Our core ResNet50 computer vision model was evaluated on the held-out validation set (17,572 images) after two phases of training (initial frozen backbone + full fine-tuning).

- **Macro-F1 Score:** `0.9703`
- **Overall Accuracy:** `97.06%`

*For full class-by-class precision/recall and the confusion matrix, please see the [1-Page Model Report](report/model_report.md).*

---

## 5. Architecture Overview & Limitations

### Architecture
- **Backend:** FastAPI (Python) serving both the API routes and the static HTML frontend.
- **Frontend:** Vanilla HTML/CSS/JS (no heavy frameworks) for maximum performance and easy deployment.
- **AI/ML:** 
  - **Vision:** TensorFlow/Keras (`tensorflow-cpu` for deployment efficiency). ResNet50 backbone initialized with ImageNet weights, modified with a custom classification head (`GlobalAveragePooling2D → Dense(128) → Dense(64) → Dense(38)`).
  - **Tabular:** Scikit-Learn `RandomForestClassifier` trained on synthetic NPK/Weather data for crop recommendation.
  - **GenAI:** Integrated via `requests` to OpenRouter (using liquid/lfm-2.5-2.6b:free by default).

### Known Limitations
- **Generalization Gap:** The model was trained and validated on lab-condition images (clean backgrounds, uniform lighting). As expected in the problem statement, accuracy degrades when presented with real-world field photos featuring clutter, variable lighting, or multiple leaves.
- **Class Confusion:** There is a known confusion cluster among Tomato diseases (Early Blight vs. Target Spot) due to visually overlapping necrotic lesion symptoms.
- **Simulated Hardware:** IoT metrics are currently simulated via a background async loop in Python. Physical ESP32/Raspberry Pi hardware would need to be integrated for real-world deployment.

---

## 6. Links & Demonstration

- **Live Deployed App:** [https://agrismart-ai-8elm.onrender.com/]
- **Demo Video (3-5 mins):** [https://drive.google.com/file/d/13iBVrmlFC4Bcu7lXGPLbkdWuBxwaoGIb/view?usp=sharing]
- **Originality Declaration:** The core ResNet50 transfer learning architecture was adapted from standard Kaggle tutorials for the PlantVillage dataset. The data augmentation strategy, fine-tuning phase, full backend integration, synthetic dataset generation for Bonus A, and the entire frontend UI are original work created for this hackathon. AI coding assistants (Claude/Gemini) were utilized to accelerate boilerplate generation and deployment configuration.