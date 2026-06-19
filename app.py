import streamlit as st
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Configure the page
st.set_page_config(page_title="Iris AI Predictor", page_icon="🌸", layout="centered")

# --- Model Training ---
@st.cache_resource
def train_model():
    iris = load_iris()
    X = iris.data.tolist()
    y = iris.target.tolist()
    target_names = list(iris.target_names)
    
    # 🌹 Add Rose (Class 3)
    X.extend([[9.0, 4.0, 8.0, 3.0], [8.5, 3.8, 7.5, 2.8], [9.2, 4.2, 8.2, 3.2]])
    y.extend([3, 3, 3])
    target_names.append('rose')
    
    # 🌻 Add Sunflower (Class 4)
    X.extend([[15.0, 8.0, 10.0, 5.0], [14.0, 7.5, 9.5, 4.8], [15.5, 8.2, 10.5, 5.2]])
    y.extend([4, 4, 4])
    target_names.append('sunflower')
    
    # 🌷 Add Tulip (Class 5)
    X.extend([[6.5, 2.5, 5.5, 2.5], [6.2, 2.2, 5.0, 2.2], [6.8, 2.8, 5.8, 2.8]])
    y.extend([5, 5, 5])
    target_names.append('tulip')

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    return clf, target_names

model, class_names = train_model()

# --- Web UI ---
st.title("🌸 Expanded AI Flower Predictor")
st.markdown("We've expanded the AI! Adjust the measurements below, and the model will predict between **Iris**, **Rose**, **Sunflower**, and **Tulip**!")

st.header("Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 20.0, 5.8)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 10.0, 3.0)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 12.0, 4.35)
    petal_width = st.slider("Petal Width (cm)", 0.1, 8.0, 1.3)

# --- Prediction ---
st.divider()

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction_idx = model.predict(features)[0]
predicted_species = class_names[prediction_idx].capitalize()

st.subheader("AI Prediction Result:")

# Output different colors based on species
if predicted_species == "Setosa":
    st.success(f"The model predicts: **{predicted_species}** 🌷")
elif predicted_species == "Versicolor":
    st.info(f"The model predicts: **{predicted_species}** 🌻")
elif predicted_species == "Virginica":
    st.warning(f"The model predicts: **{predicted_species}** 🌺")
elif predicted_species == "Rose":
    st.error(f"The model predicts: **{predicted_species}** 🌹")
elif predicted_species == "Sunflower":
    st.success(f"The model predicts: **{predicted_species}** 🌻")
elif predicted_species == "Tulip":
    st.info(f"The model predicts: **{predicted_species}** 🌷")
else:
    st.warning(f"The model predicts: **{predicted_species}** 🌸")

# --- Display Image ---
img_path = os.path.join(os.path.dirname(__file__), f"{predicted_species.lower()}.png")
if os.path.exists(img_path):
    st.image(img_path, width=300)
else:
    st.caption(f"(No image found for {predicted_species})")