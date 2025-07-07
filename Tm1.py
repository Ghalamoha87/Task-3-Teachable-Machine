import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

# Load model and class labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Predict function
def predict_image(file_path):
    image = Image.open(file_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence = prediction[0][index]
    return class_name, confidence

# GUI function
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result, score = predict_image(file_path)
        result_label.config(text=f"Type: {result}\nConfidence: {score:.2f}")

# GUI layout
root = tk.Tk()
root.title("Flower Type Recognize")
root.geometry("320x200")

upload_btn = tk.Button(root, text="Choose Image", command=upload_image)
upload_btn.pack(pady=20)

result_label = tk.Label(root, text="Upload a flower image")
result_label.pack()

root.mainloop()
