from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model


app = FastAPI(title="CIFAR-10 Model Serving API")


MODEL_PATH = Path("checkpoints/classifier_v1.pt")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(
    architecture="resnet18",
    num_classes=10,
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device),
    }


@app.get("/model-info")
def model_info():
    return {
        "architecture": "resnet18",
        "num_classes": 10,
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    class_index = predicted.item()

    return {
        "class_index": class_index,
        "class_name": CLASS_NAMES[class_index],
        "confidence": round(confidence.item(), 4),
    }