import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import timm

# --- Configuration ---
SR = 32000
DURATION = 5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
MODEL_PATH = 'best_model.pth'
RESUME = True  # Set to True to continue training
PATIENCE = 5  # For early stopping
MIN_LR = 1e-6  # Minimum learning rate

# Create logs directory
os.makedirs('logs', exist_ok=True)
writer = SummaryWriter(f"logs/birdclef_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# --- Dataset ---
class AudioAugmentation:
    def __init__(self, sr=SR, p=0.5):
        self.sample_rate = sr
        self.p = p
        
    def __call__(self, waveform):
        # Time Stretching
        if random.random() < self.p:
            rate = random.uniform(0.9, 1.1)
            waveform = torchaudio.functional.resample(
                waveform, 
                orig_freq=int(self.sample_rate * rate), 
                new_freq=self.sample_rate
            )
            
        # Add Gaussian Noise
        if random.random() < self.p/2:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            
        # Random Gain
        if random.random() < self.p/2:
            gain = random.uniform(0.5, 1.5)
            waveform = waveform * gain
            
        return waveform

class BirdSoundDataset(Dataset):
    def __init__(self, df, data_path, sr=SR, duration=DURATION, augment=False):
        self.df = df
        self.data_path = data_path
        self.sr = sr
        self.duration = duration
        self.num_samples = self.sr * self.duration
        self.augment = augment
        self.augmenter = AudioAugmentation(sr=sr) if augment else None
        
        # Spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            win_length=1024,
            hop_length=512,
            n_mels=128
        )
        
        # Time masking
        self.time_mask = T.TimeMasking(time_mask_param=20)
        # Frequency masking
        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.data_path, row['filename'])
        
        try:
            waveform, sample_rate = torchaudio.load(filepath)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Resample if needed
        if sample_rate != self.sr:
            resampler = T.Resample(sample_rate, self.sr)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or trim
        if waveform.shape[1] < self.num_samples:
            padding = self.num_samples - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.num_samples]
            
        # Apply audio augmentation
        if self.augment and self.augmenter:
            waveform = self.augmenter(waveform)

        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Apply time and frequency masking for augmentation
        if self.augment:
            mel_spec = self.time_mask(mel_spec)
            mel_spec = self.freq_mask(mel_spec)
            
        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(0)  # [1, n_mels, time]
        
        label = torch.tensor(row['primary_label_encoded'], dtype=torch.long)

        return mel_spec, label

# --- Model ---
class BirdClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0'):
        super(BirdClassifier, self).__init__()
        # Use efficientnet as backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove the original classifier
            in_chans=1      # Single channel input (grayscale spectrogram)
        )
        
        # Get the number of features from the backbone
        num_features = self.backbone.num_features
        
        # Custom head
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: [batch_size, 1, n_mels, time]
        x = self.backbone(x)  # [batch_size, num_features]
        x = self.head(x)      # [batch_size, num_classes]
        return x

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

# --- Main Training Loop ---
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_metadata = pd.read_csv('train.csv')
    taxonomy = pd.read_csv('taxonomy.csv')
    
    # Create label encoding
    label_to_int = {label: i for i, label in enumerate(taxonomy['primary_label'].unique())}
    train_metadata['primary_label_encoded'] = train_metadata['primary_label'].map(label_to_int)
    train_metadata = train_metadata.dropna(subset=['primary_label_encoded'])
    
    # Get class distribution
    class_counts = train_metadata['primary_label_encoded'].value_counts().sort_index()
    print("\nClass distribution:")
    for i, count in class_counts.items():
        print(f"{taxonomy.iloc[i]['common_name']}: {count} samples")
    
    # Calculate class weights for imbalanced dataset
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Split data
    train_df, val_df = train_test_split(
        train_metadata, 
        test_size=0.2, 
        random_state=42,
        stratify=train_metadata['primary_label_encoded']
    )
    
    # Create datasets
    train_dataset = BirdSoundDataset(
        train_df, 
        data_path='train_audio', 
        sr=SR, 
        duration=DURATION,
        augment=True
    )
    
    val_dataset = BirdSoundDataset(
        val_df,
        data_path='train_audio',
        sr=SR,
        duration=DURATION,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    num_classes = len(label_to_int)
    model = BirdClassifier(num_classes=num_classes).to(device)
    
    # Load pre-trained weights if resuming
    if RESUME and os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # Keep weights_only=False for compatibility
        
        # Get the number of classes from the checkpoint
        if 'model_state_dict' in checkpoint:
            checkpoint_classes = checkpoint['model_state_dict']['head.3.weight'].shape[0]
            print(f"Checkpoint has {checkpoint_classes} classes")
            
            # Create a new model with the correct number of classes
            model = BirdClassifier(num_classes=checkpoint_classes).to(device)
            
            # Load the state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pre-trained model with {checkpoint_classes} classes")
        else:
            print("No model_state_dict found in checkpoint")
            model = BirdClassifier(num_classes=num_classes).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=True,
        min_lr=MIN_LR
    )
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Step the scheduler
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_to_int': label_to_int,
                'num_classes': model.head[3].out_features  # Save the number of output classes
            }
            torch.save(checkpoint, 'best_model.pth')
            patience_counter = 0
            print(f"Model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter}/{PATIENCE} epochs")
            
            # Early stopping
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("Training complete!")
    writer.close()
    
    # Save the final model
    final_checkpoint = {
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'label_to_int': label_to_int,
        'num_classes': model.head[3].out_features
    }
    torch.save(final_checkpoint, 'final_model.pth')
    print("Final model saved")

if __name__ == '__main__':
    # You will need to re-download train.csv and taxonomy.csv as well
    if not os.path.exists('train_audio'):
        print("Error: 'train_audio' directory not found.")
        print("Please download the dataset from https://www.kaggle.com/competitions/birdclef-2025/data")
    elif not os.path.exists('train.csv') or not os.path.exists('taxonomy.csv'):
        print("Error: 'train.csv' or 'taxonomy.csv' not found.")
        print("Please download them from the Kaggle competition page.")
    else:
        main()
