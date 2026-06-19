import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import os
import sys
from PIL import Image, ImageTk

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# 1. Train the Machine Learning Model
def train_model():
    # Load the famous Iris dataset
    iris = load_iris()
    X = iris.data.tolist()
    y = iris.target.tolist()
    target_names = list(iris.target_names)
    
    # 🌹 Add Rose
    X.extend([[9.0, 4.0, 8.0, 3.0], [8.5, 3.8, 7.5, 2.8], [9.2, 4.2, 8.2, 3.2]])
    y.extend([3, 3, 3])
    target_names.append('rose')
    
    # 🌻 Add Sunflower
    X.extend([[15.0, 8.0, 10.0, 5.0], [14.0, 7.5, 9.5, 4.8], [15.5, 8.2, 10.5, 5.2]])
    y.extend([4, 4, 4])
    target_names.append('sunflower')
    
    # 🌷 Add Tulip
    X.extend([[6.5, 2.5, 5.5, 2.5], [6.2, 2.2, 5.0, 2.2], [6.8, 2.8, 5.8, 2.8]])
    y.extend([5, 5, 5])
    target_names.append('tulip')

    # Initialize and train a Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    return clf, target_names

# Train the model upon startup
try:
    model, class_names = train_model()
except Exception as e:
    messagebox.showerror("Error", f"Failed to train the model: {e}")
    exit()

# 2. Build the Graphical User Interface (GUI)
def predict_species():
    try:
        # Get values from the sliders
        sepal_length = float(sl_slider.get())
        sepal_width = float(sw_slider.get())
        petal_length = float(pl_slider.get())
        petal_width = float(pw_slider.get())
        
        # Make a prediction using the trained model
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction_index = model.predict(features)[0]
        predicted_class = class_names[prediction_index]
        
        # Display the result
        result_label.config(text=f"Predicted Species: {predicted_class.capitalize()}", fg="green")
        
        # Load and display the image
        img_path = resource_path(f"{predicted_class}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.resize((150, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            image_label.config(image=photo)
            image_label.image = photo  # keep a reference!
        else:
            image_label.config(image='', text="Image not found")
            
    except ValueError:
        result_label.config(text="Please enter valid numbers.", fg="red")

# Create the main window
root = tk.Tk()
root.title("Iris Flower AI Classifier")
root.geometry("400x550")
root.configure(padx=20, pady=20)

# Title Label
title_label = tk.Label(root, text="Iris Flower AI Classifier", font=("Helvetica", 16, "bold"))
title_label.pack(pady=(0, 10))

instructions_label = tk.Label(root, text="Adjust the flower measurements to predict its species.", font=("Helvetica", 10))
instructions_label.pack(pady=(0, 15))

# Input Frame for sliders
input_frame = tk.Frame(root)
input_frame.pack()

# Sliders for the 4 features
tk.Label(input_frame, text="Sepal Length (cm):").grid(row=0, column=0, sticky="e")
sl_slider = tk.Scale(input_frame, from_=4.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL)
sl_slider.set(5.8)
sl_slider.grid(row=0, column=1)

tk.Label(input_frame, text="Sepal Width (cm):").grid(row=1, column=0, sticky="e")
sw_slider = tk.Scale(input_frame, from_=2.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL)
sw_slider.set(3.0)
sw_slider.grid(row=1, column=1)

tk.Label(input_frame, text="Petal Length (cm):").grid(row=2, column=0, sticky="e")
pl_slider = tk.Scale(input_frame, from_=1.0, to=12.0, resolution=0.1, orient=tk.HORIZONTAL)
pl_slider.set(4.35)
pl_slider.grid(row=2, column=1)

tk.Label(input_frame, text="Petal Width (cm):").grid(row=3, column=0, sticky="e")
pw_slider = tk.Scale(input_frame, from_=0.1, to=8.0, resolution=0.1, orient=tk.HORIZONTAL)
pw_slider.set(1.3)
pw_slider.grid(row=3, column=1)

# Predict Button
predict_btn = tk.Button(root, text="Predict Species", command=predict_species, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
predict_btn.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="Prediction will appear here.", font=("Helvetica", 12, "bold"))
result_label.pack()

# Image Label
image_label = tk.Label(root)
image_label.pack(pady=10)

# Run the application
root.mainloop()
