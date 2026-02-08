# Disease Prediction from Symptoms (ML Project)

This is a small machine learning project where I built a model to predict possible diseases based on reported symptoms.

The goal was mainly to practice:
- Data preprocessing
- Handling class imbalance
- Multi-class classification
- Model evaluation
- Saving/loading ML models
- Creating a simple interactive interface

This is a learning project and not a medical tool.

---

## ⚠️ Disclaimer

This project is for educational purposes only.  
It should NOT be used for medical diagnosis or health decisions.

---

##  About the Dataset

The dataset contains symptom–disease mappings where:
- Symptoms are binary (0/1)
- Each row represents a symptom combination
- Target is the disease (prognosis)

It is a clean and well-structured dataset often used for ML practice.

---

##  What I Did
- Prepared the dataset
- Encoded disease labels using LabelEncoder
- Used **SMOTE** to handle class imbalance
- Trained an **XGBoost classifier**
- Evaluated performance using:
  - Accuracy
  - Classification report
  - Cross-validation
- Saved the model and encoder using joblib
- Built a small interactive widget for predictions

---

## 📊 Results

The model achieves very high accuracy on this dataset.

This likely reflects the clean and separable nature of the data.  
Real-world medical data would be noisier and more complex.

---

## ▶️ How to Run

1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Open the notebook and run all cells

---

## Limitations

- Dataset is simplified compared to real clinical data  
- Predictions depend fully on symptom input  
- No real patient validation  
- Not production-ready yet

---

##  Future Improvements

Some ideas I may explore later:
- Deploy as a Streamlit app
- Add explainability (SHAP/LIME)
- Try different models
- Use more realistic datasets

---

## 🙌 Final Note

This project was built as part of my learning journey in AI/ML.  
Feedback or suggestions are always welcome.


