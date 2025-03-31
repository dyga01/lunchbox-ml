# development notes

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

```text
lunchbox --help
lunchbox train --model test_models.lstm_model
lunchbox test --model test_models.gru_model
lunchbox deploy --model test_models.cnn_model
```
