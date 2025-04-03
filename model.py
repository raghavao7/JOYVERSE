import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import Dataset, DataLoader

# Define the LandmarkEmotionTransformer
class LandmarkEmotionTransformer(nn.Module):
    def __init__(self, num_landmarks=468, num_emotions=6, d_model=128, nhead=8, num_layers=4):
        super(LandmarkEmotionTransformer, self).__init__()
        self.num_landmarks = num_landmarks
        self.d_model = d_model
        self.emotions = ["happy", "disgust", "sad", "angry", "neutral", "fear"]
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}

        self.input_embedding = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(num_landmarks * d_model, num_emotions)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, landmarks):
        batch_size = landmarks.size(0)
        x = self.input_embedding(landmarks)
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, -1)
        logits = self.fc(x)
        return logits

# Custom Dataset Class for Excel
class FacialLandmarkDataset(Dataset):
    def __init__(self, excel_file, scaler=None):
        self.data = []
        self.labels = []
        self.emotion_to_idx = {"happy": 0, "disgust": 1, "sad": 2, "angry": 3, "neutral": 4, "fear": 5}

        # Load the Excel file
        df = pd.read_excel(excel_file)
        print("Excel file loaded. First few rows:")
        print(df.head())  # Debug: Show the first few rows

        # Ensure the expected columns exist
        expected_columns = ["Expression"] + [f"{coord}_{i}" for i in range(468) for coord in ["x", "y", "z"]]
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Excel file is missing columns: {missing_cols}")

        # Process each row
        for idx, row in df.iterrows():
            emotion = row["Expression"]
            if pd.isna(emotion):  # Skip rows with NaN in "Expression"
                print(f"Skipping row {idx}: 'Expression' is NaN")
                continue
            if not isinstance(emotion, str):  # Ensure emotion is a string
                print(f"Skipping row {idx}: 'Expression' is not a string ({emotion})")
                continue
            emotion = emotion.lower()
            if emotion not in self.emotion_to_idx:
                print(f"Skipping row {idx}: Unknown emotion '{emotion}'")
                continue

            landmarks = []
            try:
                for i in range(468):
                    x = float(row[f"x_{i}"])
                    y = float(row[f"y_{i}"])
                    z = float(row[f"z_{i}"])
                    landmarks.extend([x, y, z])
                self.data.append(landmarks)
                self.labels.append(self.emotion_to_idx[emotion])
            except (ValueError, KeyError) as e:
                print(f"Skipping row {idx}: Invalid landmark data ({str(e)})")
                continue

        if not self.data:
            raise ValueError("No valid data found in the Excel file after processing.")

        self.data = np.array(self.data, dtype=np.float32).reshape(-1, 468, 3)  # Shape: (samples, 468, 3)
        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"Processed {len(self.data)} valid samples.")

        # Normalize the data
        if scaler is None:
            self.scaler = StandardScaler()
            data_reshaped = self.data.reshape(-1, 3)  # Flatten to (samples * 468, 3)
            self.data = self.scaler.fit_transform(data_reshaped).reshape(self.data.shape)
            joblib.dump(self.scaler, "scaler.pkl")
        else:
            self.scaler = scaler
            data_reshaped = self.data.reshape(-1, 3)
            self.data = self.scaler.transform(data_reshaped).reshape(self.data.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Training Function
def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for landmarks, labels in train_loader:
            landmarks, labels = landmarks.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "emotion_detector.pth")
    print("Model saved as 'emotion_detector.pth'")

# Main Execution
if __name__ == "__main__":
    # Path to your Excel file
    excel_file = "JoyVerseDataSet_Filled.xlsx"  # Adjust path as needed

    # Load dataset
    dataset = FacialLandmarkDataset(excel_file)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = LandmarkEmotionTransformer(num_landmarks=468, num_emotions=6)

    # Train the model
    train_model(model, train_loader, num_epochs=10)

    print("Training completed!")