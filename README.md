
# 🛰️ Satellite Imagery-Based Property Valuation

## Overview

This project builds a **multimodal regression system** to predict residential property prices by combining:

* **Tabular housing data** (size, rooms, location, etc.)
* **Satellite imagery** capturing neighborhood and environmental context

The goal is to demonstrate how **visual features such as greenery, road density, and urban layout** can improve traditional real-estate valuation models.

---

## Dataset

* **Tabular Data:** King County Housing Dataset
* **Target Variable:** `price`
* **Key Features:**
  `bedrooms`, `bathrooms`, `sqft_living`, `lat`, `long`, and neighborhood statistics
* **Visual Data:**
  Satellite images fetched using **Mapbox Static Images API** based on latitude and longitude

---

## Project Structure

```
.
├── data/
│   ├── raw/                  # Original train & test files
│   ├── processed/             # Cleaned datasets
│   ├── images/                # Satellite images (train/test)
│   └── embeddings/            # CNN image embeddings
│
├── data_fetcher.py            # Script to download satellite images
├── preprocessing.ipynb        # Data cleaning & EDA
├── model_training.ipynb       # Model training, evaluation & inference
├── predictions.csv            # Final test set predictions
└── README.md
```

---

## Methodology

1. **Exploratory Data Analysis (EDA)**
   Analyzed price distributions, feature correlations, and spatial patterns.

2. **Satellite Image Acquisition**
   Programmatically downloaded satellite images using property coordinates.

3. **Image Feature Engineering**
   Used a pretrained **ResNet-18 CNN** to extract 512-dimensional image embeddings.

4. **Multimodal Fusion**
   Concatenated tabular features with image embeddings.

5. **Model Training**
   Trained a **Random Forest Regressor** on combined features.

6. **Explainability**
   Applied **Grad-CAM** to highlight influential regions in satellite images.

---

## Results

| Model                         | RMSE      | MAE      | R²       |
| ----------------------------- | --------- | -------- | -------- |
| Tabular Only                  | ~210k     | ~131k    | 0.65     |
| Multimodal (Tabular + Images) | **~149k** | **~79k** | **0.83** |

Satellite imagery significantly improved prediction accuracy.

---

## Prediction File

The final predictions for the test set are stored in:

```
predictions.csv
```

Format:

```
id,predicted_price
```

---

## How to Run

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn torch torchvision matplotlib
```

2. Download satellite images:

```bash
python data_fetcher.py
```

3. Run notebooks in order:

* `preprocessing.ipynb`
* `model_training.ipynb`

---

## Key Takeaways

* Environmental context plays a major role in property valuation.
* Multimodal learning outperforms traditional tabular-only approaches.
* Satellite imagery provides interpretable and valuable spatial signals.

---

## Future Work

* Fine-tuning CNN layers
* Using higher-resolution imagery
* Incorporating street-view or temporal data

---

## Author

**Shruti (23411036)**

