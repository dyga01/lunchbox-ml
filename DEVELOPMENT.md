# development notes

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

```text
lunchbox --help
lunchbox train --model ./test_models/gru_model.py
lunchbox train --model ./test_models/gru_model.py --output
lunchbox train --model ./test_models/gru_model.py --output --benchmark
```

```text
lunchbox serve --model ./test_models/gru_model.pth --backend pytorch
lunchbox serve --model ./test_models/gru_model.onnx --backend onnx
lunchbox serve --model ./test_models/gru_model.pth --backend torchserve
lunchbox serve --model ./test_models/gru_model.mlmodel --backend coreml
```

## Final tasks

- implement train functionality, explore using mojo for optimizations
- implement deploy functionality by automatically generating a dockerfile?
- test lastly

- make the tool useable, this may include a lot of code that will automatically install models dependencies and logic around this --> make a setup shell script for this?
- error handling
- documentation of everything
- linting with ruff
- logo
- finalized readme and example commands
- make it public?
- license in repo?
- implement serve functionality for different backends (pytorch, onnx, torchserve, coreml)

## TODO LIST

- run multiple benchmarks and display results
- serve the model in different ways --> pytorch serve, onnx runtime serve, more
