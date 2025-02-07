from fastapi import FastAPI, File, UploadFile, Form
import torch
from io import BytesIO
from PIL import Image
import json
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import os

app = FastAPI()

embedding_file_path = "embeddings.json"

if not os.path.exists(embedding_file_path):
    with open(embedding_file_path, "w") as f:
        json.dump({}, f)

class StudentSiameseNetwork(nn.Module):
    def __init__(self, embedding_size=128):
        super(StudentSiameseNetwork, self).__init__()
        efficientnet = models.efficientnet_b0(pretrained=True)
        for param in efficientnet.parameters():
            param.requires_grad = False

        self.backbone = efficientnet.features
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)
        )

    def forward(self, input):
        x = self.backbone(input)
        x = self.fc(x)
        return x

model = StudentSiameseNetwork(embedding_size=128)
model.load_state_dict(torch.load("Student_facial_recognition_model.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embedding(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze().cpu().numpy()

def cosine_similarity(embedding1, embedding2):
    tensor1 = torch.tensor(embedding1).unsqueeze(0)
    tensor2 = torch.tensor(embedding2).unsqueeze(0)
    similarity = F.cosine_similarity(tensor1, tensor2)
    return similarity.item()

@app.post("/register/")
async def register_person(file: UploadFile = File(...), name: str = Form(...)):
    image_bytes = await file.read()
    new_embedding = get_embedding(image_bytes)

    try:
        with open(embedding_file_path, "r") as f:
            embeddings_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        embeddings_dict = {}

    if name in embeddings_dict:
        try:
            old_embedding = np.array(embeddings_dict[name])
            new_embedding = (old_embedding + new_embedding) / 2
            message = f"✅ Embedding updated for {name}"
        except Exception as e:
            return {"error": f"⚠️ Failed to update embedding: {str(e)}"}
    else:
        message = f"Registered"

    embeddings_dict[name] = new_embedding.tolist()

    try:
        with open(embedding_file_path, "w") as f:
            json.dump(embeddings_dict, f, indent=4)
        return {"message": message}
    except Exception as e:
        return {"error": f"❌ Failed to save embedding: {str(e)}"}

@app.post("/recognize/")
async def recognize_person(file: UploadFile = File(...)):
    image_bytes = await file.read()
    uploaded_embedding = get_embedding(image_bytes)

    try:
        with open(embedding_file_path, "r") as f:
            embeddings_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"message": "⚠️ Not recognized"}

    max_similarity = 0.0
    best_match = None

    for name, stored_embedding in embeddings_dict.items():
        sim = cosine_similarity(uploaded_embedding, np.array(stored_embedding))
        if sim > max_similarity:
            max_similarity = sim
            best_match = name

    if max_similarity > 0.7:
        return {"message": "✅ Recognized", "recognized_user": best_match}
    else:
        return {"message": "⚠️ Not recognized"}
