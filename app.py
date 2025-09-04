from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
import os
import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import timm

# --- Model & Inference Configuration ---
SR = 32000
DURATION = 5
MODEL_PATH = 'best_model.pth'

# --- Custom Model Definition ---
class CustomBirdModel(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0'):
        super().__init__()
        # Load pre-trained EfficientNet backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,  # Remove the original classifier
            in_chans=1      # Single channel input (grayscale spectrogram)
        )
        
        # Get the number of features from the backbone
        num_features = self.backbone.num_features
        
        # Custom head to match the saved model's structure
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# --- Load Model and Taxonomy ---
def load_model_and_taxonomy():
    # The model was trained on a specific subset of 10 birds in this exact order
    # This order must match the training script's label encoding
    target_species = [
        'amakin1',  # Amazon Kingfisher
        'amekes',   # American Kestrel
        'ampkin1',  # American Pygmy Kingfisher
        'anhing',   # Anhinga
        'babwar',   # Bay-breasted Warbler
        'bafibi1',  # Bare-faced Ibis
        'banana',   # Bananaquit
        'baymac',   # Blue-and-yellow Macaw
        'bbwduc',   # Black-bellied Whistling-Duck
        'bicwre1'   # Bicolored Wren
    ]
    
    # Read taxonomy and filter for our target species
    taxonomy = pd.read_csv('taxonomy.csv')
    taxonomy = taxonomy[taxonomy['primary_label'].isin(target_species)]
    
    # Ensure the species are in the correct order by reindexing
    taxonomy = taxonomy.set_index('primary_label').loc[target_species].reset_index()
    num_classes = len(taxonomy)
    
    # Create a mapping from model's output index to bird info
    label_info = {
        i: (row['primary_label'], row['common_name']) 
        for i, row in taxonomy.iterrows()
    }
    
    # Instantiate the model with the correct architecture
    model = CustomBirdModel(num_classes=num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # Load the state dict (it should have 'model_state_dict' key)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    return model, label_info

model, label_info = load_model_and_taxonomy()

# --- Prediction Function ---
def get_prediction(audio_path):
    num_samples = SR * DURATION
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != SR:
        resampler = torchaudio.transforms.Resample(sample_rate, SR)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if waveform.shape[1] < num_samples:
        padding = num_samples - waveform.shape[1]
        waveform = nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :num_samples]

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(SR)(waveform)
    # The model expects a 3-channel input, so we stack the single-channel spectrogram
    # The model expects a [batch_size, channels, height, width] input.
    # Our spectrogram is [1, H, W], so we just need to add the batch dimension.
    mel_spectrogram = mel_spectrogram.unsqueeze(0) # From [1, H, W] to [1, 1, H, W]

    with torch.no_grad():
        outputs = model(mel_spectrogram)
        print(f"Raw model outputs: {outputs}")
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        print(f"Probabilities: {probabilities}")
        
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Print the label mapping for debugging
    print("\nLabel mapping:")
    for idx, (sci_name, common_name) in label_info.items():
        print(f"Index {idx}: {common_name} ({sci_name})")
    
    predictions = []
    for i in range(len(top5_indices)):
        idx = top5_indices[i].item()
        scientific_name, common_name = label_info[idx]
        confidence = top5_prob[i].item()
        print(f"\nPrediction {i+1}:")
        print(f"  Index: {idx}")
        print(f"  Species: {common_name}")
        print(f"  Scientific Name: {scientific_name}")
        print(f"  Confidence: {confidence:.2f}")
        
        predictions.append({
            "species": common_name,
            "scientific_name": scientific_name,
            "confidence": confidence,
            "image_url": "/static/images/placeholder-bird.jpg"
        })
        
    return predictions


app = FastAPI(title="BirdCLEF 2025 - Bird Sound Classifier")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the project directory
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create necessary directories if they don't exist
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)
(STATIC_DIR / "images").mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{audio.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await audio.read())
        
        # Get predictions from the model
        predictions = get_prediction(temp_path)
        
        # Clean up
        os.remove(temp_path)
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
