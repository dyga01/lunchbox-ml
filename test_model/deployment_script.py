
import torch
import torch.nn as nn
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Define the model architecture
class AirbnbRatingPredictor(nn.Module):
    def __init__(self, input_dim, hidden_size_1, hidden_size_2, hidden_size_3, dropout_rate):
        super(AirbnbRatingPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size_2),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load model and scaler
model_path = "airbnb_rating_predictor.pth"
scaler_path = "feature_scaler.pkl"

model = AirbnbRatingPredictor(input_dim=6, hidden_size_1=128, hidden_size_2=64, hidden_size_3=32, dropout_rate=0.2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = pd.DataFrame([data])
    input_features_normalized = scaler.transform(input_features)
    input_tensor = torch.tensor(input_features_normalized, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_rating = prediction.item()
        predicted_rating = max(0, min(5, predicted_rating))

    return jsonify({"rating": predicted_rating})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
    