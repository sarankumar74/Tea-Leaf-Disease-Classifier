
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import datetime
import numpy as np

# --- Configuration ---
CLASS_NAMES = ['Algal Spot', 'Brown Blight', 'Healthy', 'Unknown']

# --- Sidebar Settings ---
st.sidebar.markdown("## ⚙️ **Settings**")
SOFTMAX_THRESHOLD = st.sidebar.slider("Softmax Confidence Threshold", 0.5, 1.0, 0.85)
ENTROPY_THRESHOLD = st.sidebar.slider("Entropy Threshold", 0.5, 2.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** VGG16 (Pretrained on Tea Leaf Dataset)")
st.sidebar.markdown("**Classes:** Algal Spot / Brown Blight / Healthy")

# --- Page Setup ---
st.set_page_config(page_title="🌿 Tea Leaf Disease Classifier", page_icon="🍃", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
        body {background-color: #f5f9f5;}
        .title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #24543E;
            margin-bottom: 0.1em;
        }
        .subtitle {
            text-align: center;
            color: #446D52;
            font-size: 1.1em;
            font-weight: 500;
        }
        .block-container {
            padding-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    num_classes = len(CLASS_NAMES)
    model = models.vgg16(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    state_dict = torch.load("Tea_Leaf_Disease.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Helper Functions ---
def softmax_entropy(probs):
    """Entropy measure for uncertainty"""
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    return entropy

def looks_like_leaf(image):
    """Heuristic to reject non-leaf images (checks green color dominance)"""
    img_array = np.array(image)
    avg_rgb = img_array.mean(axis=(0, 1))
    green_ratio = avg_rgb[1] / (avg_rgb[0] + avg_rgb[2] + 1e-8)
    return green_ratio > 0.35  

# --- App Header ---
st.markdown(""" <h1 style="text-align:center; color:#24543E; font-weight:900;">🌿 Tea Leaf Disease Classifier </h1>
    <h3 style="text-align:center; color:#446D52; font-weight:700;">
        Upload a tea leaf image to detect Algal Spot, Brown Blight, or Healthy 🍃
    </h3>
""", unsafe_allow_html=True)
# --- File Upload ---
uploaded_file = st.file_uploader("📤 **Upload a Tea Leaf Image**", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # --- Two Columns Layout ---
    col1, col2 = st.columns([1.2, 1])

    
    with col1:
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

   
    with col2:

        st.markdown("### 🔎 **Prediction Result**")

        # --- Check if it looks like a leaf ---
        if not looks_like_leaf(image):
            st.error("🚫 This image doesn't appear to be a tea leaf.\nPlease upload a clear photo of a leaf.")
        else:
            # --- Model Inference ---
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                max_prob, predicted_idx = torch.max(probs, 0)
                entropy = softmax_entropy(probs)

            # --- Out-of-Distribution Check ---
            if max_prob.item() < SOFTMAX_THRESHOLD or entropy > ENTROPY_THRESHOLD:
                st.warning("⚠️ Uncertain prediction — this image might not match trained categories.")
            else:
                predicted_class = CLASS_NAMES[predicted_idx.item()]
                st.success(f"✅ **Prediction:** {predicted_class}")

