import streamlit as st
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator



class Config:
    PAGE_TITLE = "ApisTox"
    PAGE_ICON = "🐝"
    LAYOUT = "wide"


# Load Model
class ModelLoader:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return joblib.load(self.model_path)

    def home_page(self):
        st.title("🐝 ApisTox")

        st.markdown("""
### Project Overview
                    
This machine learning application predicts whether a **chemical substance is toxic to bees**.

Bees are critical pollinators responsible for supporting ecosystems and agriculture. However, exposure to agrochemicals such as pesticides, herbicides, and fungicides can negatively affect bee populations.                  

                    
### Bee Toxicity Prediction System

ApisTox predicts whether an agrochemical is toxic
to honey bees using:

- Molecular fingerprints
- Agrochemical categories
- Toxicity exposure type
- Source information

The system is powered by Machine Learning and
Morgan Fingerprints generated from SMILES strings.
                    """)


class FingerprintEncoder:

    def __init__(self):
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048
        )

    def smiles_to_fp(self, smiles):

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        fp = self.generator.GetFingerprint(mol)

        return list(fp)




class SideBar:

    def about_page(self):
        st.title("About This Project")
        st.markdown("""
### Apistox - Bee Toxicity Predictor
This project demonstrates how **machine learning can be applied to environmental risk analysis**.

The application predicts whether chemicals are **toxic to bees**, which are essential pollinators in agriculture.
                    
### Technologies Used
- Python
- Scikit-Learn
- Pandas
- Streamlit
                    
Dataset contains agrochemicals collected from:

- PPDB
- ECOTOX
- BPDB

Target Variable:

- 0 → Non Toxic
- 1 → Toxic

Features:

- Agrochemical categories
- Toxicity type
- Source
- Morgan fingerprints
                    
### Goal
To explore **machine learning applications in ecological and environmental science.**
""")
        

    def method(self):
        st.title("⚙️ Methodology")

        st.markdown("""
### Workflow

1. Data Collection
2. Data Cleaning
3. One-Hot Encoding
4. SMILES Conversion
5. Morgan Fingerprint Generation
6. Model Training
7. Toxicity Prediction

### Algorithms Evaluated

- Logistic Regression
- Random Forest

### Model Selection

GridSearchCV was used for
hyperparameter optimization.
        """)

    
    def visualize(self):
        st.title("📊 Exploratory Analysis")

        df = pd.read_csv("data/dataset_final.csv")

        st.subheader("Target Distribution")

        st.bar_chart(df["label"].value_counts())

        st.subheader("Dataset Preview")

        st.dataframe(df.head())


    def performance_page(self):

        st.title("🏆 Model Performance")

        st.metric(
            label="Accuracy",
            value="87.92%"
        )

        st.markdown("""
        ### Classification Report

        Class 0:
        - Precision: 0.87
        - Recall: 0.97

        Class 1:
        - Precision: 0.90
        - Recall: 0.64

        Weighted F1 Score:
        - 0.87
        """)

        st.success(
            "Random Forest was selected as the final model."
        )


    def fingerprint_page(self):

        st.title("🧪 Molecular Fingerprint Explorer")

        st.write("Morgan fingerprints convert chemical structures into machine-learning-readable numerical vectors.")

        smiles = st.text_input(
            "Enter SMILES"
        )

        if smiles:

            encoder = FingerprintEncoder()

            fp = encoder.smiles_to_fp(smiles)

            if fp is None:
                st.error("Invalid SMILES")
            else:

                st.write(
                    f"Fingerprint Length: {len(fp)}"
                )
                st.write(
                    f"Active bits: {sum(fp)}"
                )

    def predict(self, model):
        
        st.title("🔬 Predict Toxicity")

        st.subheader("Chemical Information")

        smiles = st.text_area(
            "SMILES String"
        )

        source = st.selectbox(
            "Source",
            ["BPDB", "ECOTOX", "PPDB"]
        )

        tox_type = st.selectbox(
            "Toxicity Type",
            ["Contact", "Oral", "Other"]
        )


        herbicide = st.selectbox(
            "Herbicide",
            [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No"
        )

        fungicide = st.selectbox(
            "Fungicide",
            [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No"
        )

        insecticide = st.selectbox(
            "Insecticide",
            [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No"
        )

        other = st.selectbox(
            "Other Agrochemical",
            [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No"
        )

        if st.button("Predict"):

            encoder = FingerprintEncoder()

            fp = encoder.smiles_to_fp(smiles)

            if fp is None:

                st.error("Invalid SMILES")
                return
            
            data = {
                "herbicide": herbicide,
                "fungicide": fungicide,
                "insecticide": insecticide,
                "other_agrochemical": other,

                "source_BPDB": int(source == "BPDB"),
                "source_ECOTOX": int(source == "ECOTOX"),
                "source_PPDB": int(source == "PPDB"),

                "toxicity_type_Contact":
                    int(tox_type == "Contact"),

                "toxicity_type_Oral":
                    int(tox_type == "Oral"),

                "toxicity_type_Other":
                    int(tox_type == "Other")
            }

            for i in range(2048):
                data[f"SMILES_fp_{i}"] = fp[i]

            X = pd.DataFrame([data])

            prediction = model.predict(X)[0]

            if prediction == 1:

                st.error(
                    "⚠️ Toxic To Honey Bees"
                )

            else:

                st.success(
                    "✅ Non-Toxic To Honey Bees"
                )



class ApisToxApp:

    def __init__(self):

        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout=Config.LAYOUT
        )

        self.model = ModelLoader(
            "apistox_model.pkl"
        ).model

    def run(self):
        app = ModelLoader("apistox_model.pkl")
        nav = SideBar()

        def predict_page():
            nav.predict(self.model)


        home = st.Page(app.home_page, title="Home", icon=":material/home:", default=True)
        about = st.Page(nav.about_page, title="About", icon=":material/info:")
        methodology = st.Page(nav.method, title="Methodology", icon=":material/flowchart:")
        visualization = st.Page(nav.visualize, title="Visualization", icon=":material/bar_chart:")
        performance = st.Page(nav.performance_page, title="Model Performance", icon=":material/trending_up:")
        fingerprint = st.Page(nav.fingerprint_page, title="Chemical Fingerprint", icon=":material/labs:")
        predict = st.Page(predict_page, title="Predict Toxicity", icon=":material/biotech:")

        pg = st.navigation([home, about, methodology, visualization, performance, fingerprint, predict])
        pg.run()

if __name__ == "__main__":
    ApisToxApp().run()