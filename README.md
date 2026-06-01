# 🐝 ApisTox: Bee Toxicity Prediction System

ApisTox is a machine learning-powered web application that predicts whether an agrochemical is toxic to honey bees using molecular fingerprints and chemical metadata.

## Overview

Honey bees play a critical role in pollination and agricultural productivity. Exposure to pesticides and other agrochemicals can negatively impact bee populations and ecosystem health.

ApisTox leverages cheminformatics and machine learning to classify chemical compounds as:

* ✅ Non-Toxic to Honey Bees
* ⚠️ Toxic to Honey Bees

The application is deployed using Streamlit and provides an interactive interface for toxicity prediction.

---

## Features

* Chemical toxicity prediction
* Morgan fingerprint generation from SMILES strings
* Interactive Streamlit dashboard
* Model performance reporting
* Exploratory data analysis page
* Feature engineering demonstration

---

## Dataset Sources

The dataset was compiled from publicly available environmental and chemical databases:

* PPDB (Pesticide Properties DataBase)
* ECOTOX
* BPDB

---

## Machine Learning Pipeline

### Data Preprocessing

* Data cleaning
* One-hot encoding of categorical features
* Molecular fingerprint generation using RDKit

### Feature Engineering

Chemical structures represented as SMILES strings are converted into:

* Morgan Fingerprints
* Radius = 2
* Fingerprint size = 2048 bits

### Algorithms Evaluated

* Logistic Regression
* Random Forest Classifier

### Model Selection

Hyperparameter tuning was performed using GridSearchCV.

The final deployed model is a Random Forest Classifier.

---

## Technologies Used

* Python
* Streamlit
* Scikit-Learn
* Pandas
* RDKit
* Joblib

---

## Project Structure

apistox/

├── app.py

├── apistox_model.pkl

├── data/

│ └── dataset_final.csv

├── apistox.ipynb

├── requirements.txt

└── README.md

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/apistox.git

cd apistox

Create a virtual environment:

python -m venv .venv

Activate the environment:

Windows:

.venv\Scripts\activate

Linux/Mac:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

## Running the Application

Start the Streamlit app:

streamlit run app.py

The application will open in your browser.

---

## Author

Deborah Okonkwo

Machine Learning & AI Enthusiast
