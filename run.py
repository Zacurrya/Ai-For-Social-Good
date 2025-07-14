'''
Image Food Nutritional Analysis
Takes a photo, analyses the food in the image, guesses what ingredients it's comprised of,
and returns a nutritional score
SDGs 3, 4, 10
'''

import os
import cv2
import torch
import requests
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# === Setup Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'resnet18_food101.pth')
CLASS_TXT_PATH = os.path.join(BASE_DIR, 'food101_classes.txt')
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")  # 🔐 Replace with your key

# === Load Class Names ===
with open(CLASS_TXT_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# === Claude API Call ===
def get_health_info(food_name: str, api_key: str) -> str:
    system_prompt = """
    You are a food health scoring assistant. 
    For every food item I give you, return a short health evaluation using this format:

    - Food: <food name>
    - Calories: <estimate>
    - Health Score: <score from 1 to 10>
    - Summary: <1-sentence health summary>

    Make sure the format is always the same and the score is based on general nutrition 
    (lower = less healthy, higher = more healthy).
    """
    user_prompt = f"{food_name}"

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 300,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            }
        )
        result = response.json()
        return result['content'][0]['text'].strip()
    except Exception as e:
        return f"⚠️ Error retrieving health info: {e}"

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

# === Define Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load Model ===
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found at:", MODEL_PATH)
    exit()

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 101)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)
print("✅ Model loaded and ready.")

# === Open Camera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Could not open webcam.")
    exit()
print("📷 Camera is running. Press 'q' to quit.")

threshold = 0.5
last_label = None
health_cache = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        max_prob, pred = torch.max(prob, 1)

    if max_prob.item() >= threshold:
        label = class_names[pred.item()]
        conf = max_prob.item()

        # Get health info from Claude once per unique label
        if label != last_label:
            if label in health_cache:
                health_info = health_cache[label]
            else:
                print(f"🤖 Querying Claude for: {label}")
                health_info = get_health_info(label, CLAUDE_API_KEY)
                health_cache[label] = health_info
            last_label = label

        # Draw food label
        cv2.rectangle(frame, (30, 30), (500, 100), (0, 255, 0), -1)
        cv2.putText(frame, f"{label} ({conf:.2f})", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        # Draw health info
        if health_info:
            y0 = 130
            for i, line in enumerate(health_info.splitlines()):
                cv2.putText(frame, line.strip(), (40, y0 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("🍽️ Food Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting...")

