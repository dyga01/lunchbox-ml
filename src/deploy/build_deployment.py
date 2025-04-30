from jinja2 import Template
from pathlib import Path
import os
import time
import subprocess
from .parse_yml import parse_yaml_file

def build_deployment_script(yaml_file_path, benchmark):
    """
    Generates a Python deployment script based on the YAML configuration file.

    Args:
        yaml_file_path (str): Path to the YAML configuration file.
        benchmark (bool): Whether to benchmark the script generation and execution.
    """
    # Parse the YAML file
    config = parse_yaml_file(yaml_file_path)

    # Extract the output file path from the YAML configuration
    output_file_path = config['deployment'][0]['path']

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
    app.run(host='0.0.0.0', port=8000)
    """)

    # Benchmark: Record the time it takes to build the deployment script
    start_time = time.time()

    # Render the template with the parsed configuration
    deployment_script = template.render(config=config)

    # Ensure the output directory exists
    output_path = Path(output_file_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the generated script to the output file
    with open(output_path, 'w') as file:
        file.write(deployment_script)

    build_time = time.time() - start_time

    # Print the deployment instructions
    print_deployment_instructions(output_path)

    if benchmark:
        # Benchmark: Record the time it takes to run the deployment script
        start_time = time.time()
        subprocess.run(["python", str(output_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run_time = time.time() - start_time

        # print the benchmark results
        GREEN = "\033[92m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        print(f"{BOLD}{GREEN}Benchmark Results{RESET}")
        print(f"Time to build the deployment script: {build_time:.4f} seconds")
        print(f"Time to run the deployment script: {run_time:.4f} seconds\n")

def print_deployment_instructions(output_path):
    """
    Print formatted deployment instructions for the generated script.

    Args:
        output_path (str): Path to the generated deployment script.
    """
    # ANSI color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Print deployment instructions
    print(f"\n{BOLD}{CYAN}Deployment script generated at:{RESET} {GREEN}{output_path}{RESET}")
    print(f"\n{BOLD}{CYAN}To run the deployment, use the following command:{RESET}")
    print(f"{BOLD}python {output_path}{RESET}")
    print(f"\n{BOLD}{CYAN}To query the deployed model, use the following command:{RESET}")
    print(f"""{BOLD}curl -X POST http://127.0.0.1:8000/predict \\
-H "Content-Type: application/json" \\
-d '{{"accommodates": 2, "bathrooms": 1, "bedrooms": 1, "beds": 1, "price": 100, "amenities_length": 10}}'{RESET}
    """)