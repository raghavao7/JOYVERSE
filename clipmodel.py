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
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# DecisionTransformer Model with GPT-2
class DecisionTransformer(nn.Module):
    def __init__(self, model_name="gpt2", action_space=3):
        super(DecisionTransformer, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.action_space = action_space
    
    def forward(self, input_sequence):
        inputs = self.tokenizer(input_sequence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}  # Move to device
        outputs = self.model(**inputs)
        logits = outputs.logits
        action = torch.argmax(logits[:, -1, :], dim=-1)  # Predict action from last token
        return action

    def predict_action(self, emotional_state, previous_actions):
        input_sequence = f"Emotion: {emotional_state} | Previous actions: {previous_actions}"
        action = self.forward(input_sequence)
        action = action.item()  # Extract scalar value
        action_map = {
            0: "Reduce Difficulty",
            1: "Increase Difficulty",
            2: "Maintain Difficulty"
        }
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
            print(f"Probabilities: {dict(zip(self.emotions, probs))}")
            return self.emotions[predicted_class_idx]
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return "neutral"

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
    
    try:
        images = [Image.open(BytesIO(frame.read())) for frame in frames if frame]
    
        # Detect emotions for all images
        detected_emotions = [emotion_detector.detect_emotion(image) for image in images]
        print(f"Detected emotions for frames: {detected_emotions}")

        # Get the dominant emotion using majority vote
        dominant_emotion = Counter(detected_emotions).most_common(1)[0][0]

        # Predict action using DecisionTransformer
        predicted_action = decision_transformer.predict_action(dominant_emotion, previous_actions)

        return jsonify({"emotion": dominant_emotion, "action": predicted_action})
    
    except Exception as e:
        print(f"Error processing frames: {str(e)}")
        return jsonify({"error": "Error processing frames"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
