# development notes

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

training models

```text
lunchbox --help
lunchbox train --model ./test_model/main.py
lunchbox train --model ./test_model/main.py --output
lunchbox train --model ./test_model/main.py --output --benchmark
```

deploying models

x

## Final tasks

- run multiple benchmarks and display results
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

- make my example model be able to be deployed
- think about the inputs and outputs and make a diagram!!
- follow new project structure to get base working. make it benchmarkable.
