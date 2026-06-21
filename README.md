
# ExoTerra — A World Away: AI for Exoplanet Detection

## Overview
ExoTerra is a machine learning system designed to identify and classify exoplanet candidates from astronomical time-series data (light curves). The project applies supervised learning techniques to detect transit-like signals that indicate the possible presence of exoplanets.

Built for NASA Space Apps Challenge, the system focuses on scalable analysis of large space datasets and fast inference through a web-integrated pipeline.

---

## Core Objective
Develop an AI-driven pipeline that:
- Processes stellar light curve data
- Extracts meaningful astrophysical features
- Detects potential exoplanet transits
- Outputs classification results via a web interface

---

## System Architecture

### 1. Machine Learning Layer
- Model: XGBoost classifier
- Input: Processed photometric / time-series features
- Output: Exoplanet probability / classification label
- Components:
  - Data cleaning and normalization
  - Feature engineering from flux signals
  - Model training and evaluation
  - Inference pipeline

---

### 2. Backend Layer
- Framework: Flask
- Responsibilities:
  - Serve trained ML model
  - Handle prediction requests via API
  - Connect frontend with ML inference engine
  - Return structured prediction results

---

### 3. Frontend Layer
- Web-based interface
- Responsibilities:
  - Accept user input or dataset selection
  - Display prediction output
  - Visualize classification results

---

## Data Pipeline
1. Raw astrophysical time-series input
2. Preprocessing (noise reduction, normalization)
3. Feature extraction (statistical + signal-based features)
4. ML model inference (XGBoost)
5. Output generation (classification result)

---

## Technologies Used
- Python
- XGBoost
- Flask
- JavaScript / HTML / CSS (frontend)
- GitHub (version control and collaboration)

---

## Team Structure

- Technical Lead / ML Integration: system design, model integration, core logic
- ML Development: model training, optimization, evaluation
- Backend Development: API + Flask integration
- Frontend Development: UI/UX implementation
- Concept & Documentation: explanation layer, diagrams, GitHub structuring
- Media & Production: video editing and presentation preparation

---

## Scientific Context
Exoplanet detection relies on identifying periodic dips in stellar brightness caused by planetary transits. This project automates detection using machine learning to reduce manual analysis of large-scale astronomical datasets.

---

## Key Features
- Automated exoplanet candidate detection
- Scalable ML inference pipeline
- Web-based interaction layer
- Modular architecture (ML + backend + frontend separation)
- Designed for real-world astronomical datasets

---

## Limitations
- Dependent on quality and completeness of input data
- Performance tied to feature engineering quality
- Not a replacement for full astrophysical validation pipelines

---

## Future Improvements
- Integration with deep learning (LSTM / Transformer models for time-series)
- Support for real NASA mission datasets (Kepler, TESS)
- Uncertainty estimation for predictions
- Real-time streaming data processing
- Improved interpretability for astrophysical validation

---
```
├── ml/
│   ├── train.py              # XGBoost training pipeline (data loading, training, evaluation)
│   ├── preprocess.py         # Data preprocessing and feature engineering
│   ├── model.pkl             # Serialized trained model for inference
│
├── backend/
│   ├── app.py                # Flask application entry point
│   ├── routes.py             # API endpoints (prediction, health checks, requests handling)
│   ├── model_loader.py       # Loads and manages trained ML model
│
├── frontend/
│   ├── index.html            # Main UI structure
│   ├── style.css             # Styling and layout
│   ├── app.js                # Frontend logic and API communication
│
├── data/
│   ├── raw/                  # Raw astronomical light curve datasets
│   ├── processed/            # Cleaned and feature-engineered datasets
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
```


---

## License
For educational and hackathon use (NASA Space Apps Challenge context).

---

## Status
Active development / Hackathon project

