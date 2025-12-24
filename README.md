# Cement Strength Prediction – End-to-End ML Project

An end-to-end Machine Learning application to predict **compressive strength of concrete** using input material composition.
The project includes **EDA, model training, MLflow experiment tracking, and Flask-based deployment**.

---

##  Project Overview
This project predicts cement strength using supervised regression models.  
It follows an industry-style ML pipeline with modular code structure, experiment tracking using **MLflow**, and cloud deployment on **Render**.

---

##  Problem Statement
To predict the **compressive strength of cement** based on its material composition such as cement, water, slag, ash, and aggregates.

---

##  Tech Stack
- **Programming Language:** Python 3.10
- **Framework:** Flask
- **ML Libraries:** Scikit-learn, NumPy, Pandas
- **Experiment Tracking:** MLflow
- **Database (optional):** MySQL (PyMySQL)
- **Deployment:** Render
- **Version Control:** Git & GitHub

---

##  Project Structure

├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── src/
│ ├── init.py
│ ├── components/
│ ├── pipelines/
│ │ ├── prediction_pipeline.py
│ │ └── training_pipeline.py
│ ├── utils.py
│ ├── exception.py
│ └── logger.py
├── templates/
│ ├── index.html
└── home.html

#### here is the link --> https://cement-strengthen-prediction-1.onrender.com/predictdata




