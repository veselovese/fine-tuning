# web_app/app.py
import gradio as gr
import torch
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as T
import numpy as np

MODEL_PATH = '../fine_tuning/models/best_model.onnx'
CLASSES_PATH = '../fine_tuning/models/classes.txt'
IMG_SIZE = 224

# Загрузка классов
with open(CLASSES_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Инициализация ONNX Runtime сессии
session = ort.InferenceSession(MODEL_PATH)

preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    if image is None:
        return "Please upload an image"
    
    # Преобразование
    input_tensor = preprocess(image).unsqueeze(0)
    
    # ONNX ожидает numpy array
    input_numpy = input_tensor.numpy()
    
    # Инференс
    inputs = {session.get_inputs()[0].name: input_numpy}
    outputs = session.run(None, inputs)
    logits = outputs[0][0]
    
    # Получение вероятностей через Softmax
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
    
    # Формирование результата
    confidences = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return confidences

# Интерфейс
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Image Classification ONNX",
    description="Upload an image to classify it using the fine-tuned model."
)

if __name__ == "__main__":
    interface.launch()