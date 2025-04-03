import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer, CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
from collections import Counter

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:*", "http://127.0.0.1:*"]}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# DecisionTransformer Model with GPT-2
class DecisionTransformer(nn.Module):
    def __init__(self, model_name="gpt2", action_space=3):
        super(DecisionTransformer, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.action_space = action_space
    
    def forward(self, input_sequence):
        inputs = self.tokenizer(input_sequence, return_tensors="pt", truncation=True, padding=True, max_length = 512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        action = torch.argmax(logits[:, -1, :], dim=-1)
        return action

    def predict_action(self, emotional_state, previous_actions):
        input_sequence = f"Emotion: {emotional_state} | Previous actions: {previous_actions}"
        action = self.forward(input_sequence)
        action = action.item()
        action_map = {0: "Reduce Difficulty", 1: "Increase Difficulty", 2: "Maintain Difficulty"}
        predicted_action = action_map.get(action, "Maintain Difficulty")
        print(f"Input Sequence: {input_sequence} | Predicted Action: {predicted_action}")
        return predicted_action

# Emotion Detection with CLIP
class EmotionDetection:
    def __init__(self):
        self.model_name = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.emotions = ["happy", "sad", "angry", "surprised", "neutral", "confused", "bored"]

    def detect_emotion(self, image):
        text_inputs = [
            "A person with a big smile, feeling happy",
            "A person expression is sad, feeling sad",
            "A person with an angry expression,big eyes, clenched jaw",
            "A person with wide eyes and raised eyebrows, feeling surprised",
            "A person with a calm, neutral expression",
            "A person with a puzzled look, feeling confused",
            "A person looking disinterested, feeling bored"
        ]
        try:
            inputs = self.processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            predicted_class_idx = probs.argmax()
            emotion_probs = dict(zip(self.emotions, probs))
            print(f"Probabilities: {emotion_probs}")
            return self.emotions[predicted_class_idx], emotion_probs
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return "neutral", {emotion: 0.0 for emotion in self.emotions}

# Initialize models
emotion_detector = EmotionDetection()
decision_transformer = DecisionTransformer(action_space=3)

@app.route("/analyze-frames", methods=["POST"])
def analyze_frames():
    frames = [request.files.get(f"frame{i}") for i in range(4)]
    if not all(frames):
        print("Missing frames")
        return jsonify({"error": "Missing frames"}), 400

    previous_actions = request.form.get("previous_actions", "neutral,question1")
    images = [Image.open(BytesIO(frame.read())) for frame in frames]
    
    # Detect emotions and probabilities for all 4 images
    detected_emotions_and_probs = [emotion_detector.detect_emotion(image) for image in images]
    detected_emotions = [item[0] for item in detected_emotions_and_probs]
    emotion_probs_list = [item[1] for item in detected_emotions_and_probs]
    print(f"Detected emotions for 4 frames: {detected_emotions}")

    # Average emotion probabilities across frames
    avg_emotion_probs = {
        emotion: sum(prob[emotion] for prob in emotion_probs_list) / len(emotion_probs_list)
        for emotion in emotion_detector.emotions
    }

    # Get the dominant emotion using majority vote
    dominant_emotion = Counter(detected_emotions).most_common(1)[0][0]

    # Predict action using DecisionTransformer
    predicted_action = decision_transformer.predict_action(dominant_emotion, previous_actions)

    return jsonify({
        "emotion": dominant_emotion,
        "emotion_probs": avg_emotion_probs,  # Return averaged probabilities
        "action": predicted_action
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
# import os
# import torch
# import torch.nn as nn
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from collections import Counter
# import joblib

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": ["http://localhost:*", "http://127.0.0.1:*"]}})

# # DecisionTransformer Model with GPT-2
# class DecisionTransformer(nn.Module):
#     def __init__(self, model_name="gpt2", action_space=3):
#         super(DecisionTransformer, self).__init__()
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token
#         self.model = GPT2LMHeadModel.from_pretrained(model_name)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.action_space = action_space
    
#     def forward(self, input_sequence):
#         inputs = self.tokenizer(input_sequence, return_tensors="pt", truncation=True, padding=True, max_length=512)
#         inputs = {key: val.to(self.device) for key, val in inputs.items()}  # Move to device
#         outputs = self.model(**inputs)
#         logits = outputs.logits
#         action = torch.argmax(logits[:, -1, :], dim=-1)  # Predict action from last token
#         return action

#     def predict_action(self, emotional_state, previous_actions):
#         input_sequence = f"Emotion: {emotional_state} | Previous actions: {previous_actions}"
#         action = self.forward(input_sequence)
#         action = action.item()  # Extract scalar value
#         action_map = {
#             0: "Reduce Difficulty",
#             1: "Increase Difficulty",
#             2: "Maintain Difficulty"
#         }
#         predicted_action = action_map.get(action, "Maintain Difficulty")
#         print(f"Input Sequence: {input_sequence} | Predicted Action: {predicted_action}")
#         return predicted_action

# # Transformer Model for Emotion Detection using Facial Landmarks
# class LandmarkEmotionTransformer(nn.Module):
#     def __init__(self, num_landmarks=468, num_emotions=6, d_model=128, nhead=8, num_layers=4):
#         super(LandmarkEmotionTransformer, self).__init__()
#         self.num_landmarks = num_landmarks
#         self.d_model = d_model
#         self.emotions = ["happy", "disgust", "sad", "angry", "neutral", "fear"]
#         self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
#         self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}

#         self.input_embedding = nn.Linear(3, d_model)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(num_landmarks * d_model, num_emotions)

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.to(self.device)

#     def forward(self, landmarks):
#         batch_size = landmarks.size(0)
#         x = self.input_embedding(landmarks)
#         x = self.transformer_encoder(x)
#         x = x.reshape(batch_size, -1)
#         logits = self.fc(x)
#         return logits

#     def predict(self, landmarks):
#         self.eval()
#         with torch.no_grad():
#             landmarks = torch.FloatTensor(landmarks).to(self.device)
#             logits = self.forward(landmarks)
#             probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
#             predicted_class_idx = probs.argmax()
#             print(f"Probabilities: {dict(zip(self.emotions, probs))}")
#             return self.emotions[predicted_class_idx]

# # Function to preprocess landmarks and predict emotion
# def predict_emotion(model, scaler, landmarks):
#     # landmarks should be a numpy array of shape (1, 468, 3)
#     landmarks_reshaped = landmarks.reshape(-1, 3)
#     landmarks_normalized = scaler.transform(landmarks_reshaped)
#     landmarks = landmarks_normalized.reshape(landmarks.shape)
#     predicted_emotion = model.predict(landmarks)
#     return predicted_emotion

# # Initialize models
# # Load the pre-trained LandmarkEmotionTransformer and scaler
# emotion_model = LandmarkEmotionTransformer(num_landmarks=468, num_emotions=6)
# emotion_model.load_state_dict(torch.load("emotion_detector.pth"))
# emotion_model.eval()  # Set to evaluation mode
# scaler = joblib.load("scaler.pkl")
# print("Emotion model and scaler loaded successfully")

# # Initialize the DecisionTransformer
# decision_transformer = DecisionTransformer(action_space=3)

# @app.route("/analyze-frames", methods=["POST"])
# def analyze_frames():
#     # Expecting 4 sets of facial landmarks in the request (as JSON)
#     try:
#         data = request.get_json()
#         if not data or "frames" not in data:
#             print("Missing or invalid frames data")
#             return jsonify({"error": "Missing or invalid frames data"}), 400

#         frames = data["frames"]  # List of 4 sets of landmarks
#         if len(frames) != 4:
#             print("Expected 4 frames, but received", len(frames))
#             return jsonify({"error": "Expected 4 frames"}), 400

#         # Each frame should be a list of 468 landmarks, each with x, y, z coordinates
#         detected_emotions = []
#         for i, frame in enumerate(frames):
#             if len(frame) != 468 or not all(len(landmark) == 3 for landmark in frame):
#                 print(f"Invalid landmark data in frame {i}")
#                 return jsonify({"error": f"Invalid landmark data in frame {i}"}), 400

#             # Convert frame to numpy array with shape (1, 468, 3)
#             landmarks = np.array([frame], dtype=np.float32)  # Shape: (1, 468, 3)
#             emotion = predict_emotion(emotion_model, scaler, landmarks)
#             detected_emotions.append(emotion)

#         print(f"Detected emotions for 4 frames: {detected_emotions}")

#         # Get the dominant emotion using majority vote
#         dominant_emotion = Counter(detected_emotions).most_common(1)[0][0]

#         # Get previous actions from the request
#         previous_actions = data.get("previous_actions", "neutral,question1")

#         # Predict action using DecisionTransformer
#         predicted_action = decision_transformer.predict_action(dominant_emotion, previous_actions)

#         return jsonify({"emotion": dominant_emotion, "action": predicted_action})

#     except Exception as e:
#         print(f"Error processing request: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)