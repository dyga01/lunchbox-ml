from jinja2 import Template
from pathlib import Path
import os
from .parse_yml import parse_yaml_file

def build_deployment_script(yaml_file_path, output_file_path):
    """
    Generates a Python deployment script based on the YAML configuration file.

    Args:
        yaml_file_path (str): Path to the YAML configuration file.
        output_file_path (str): Path to save the generated deployment script.
    """
    # Parse the YAML file
    config = parse_yaml_file(yaml_file_path)

    # Define a Jinja2 template for the deployment script
    template = Template("""
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
model_path = "{{ config['model']['trained_data_path'] }}"
scaler_path = "{{ config['model']['feature_scaler_path'] }}"

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
    app.run(host='0.0.0.0', port=5000)
    """)

    # Render the template with the parsed configuration
    deployment_script = template.render(config=config)

    # Ensure the output directory exists
    output_path = Path(output_file_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the generated script to the output file
    with open(output_path, 'w') as file:
        file.write(deployment_script)

    print(f"Deployment script generated at: {output_path}")
    print("\nTo run the deployment, use the following command:")
    print(f"python {output_path}")
    print("\nTo query the deployed model, use the following command:")
    print("""
curl -X POST http://127.0.0.1:5000/predict \\
-H "Content-Type: application/json" \\
-d '{"accommodates": 2, "bathrooms": 1, "bedrooms": 1, "beds": 1, "price": 100, "amenities_length": 10}'
    """)
