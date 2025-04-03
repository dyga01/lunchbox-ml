# development notes

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

```text
curl https://get.modular.com | sh
curl -ssL https://magic.modular.com/5cf2c264-40cf-42be-bf0f-aa05b5ab6fd1 | bash
export PATH=$HOME/.modular/bin:$PATH
echo 'export PATH=$HOME/.modular/bin:$PATH' >> ~/.zshrc
magic init optimizer --format mojoproject
cd optimizer
magic shell
mojo hello.mojo
```

```text
lunchbox --help
lunchbox train --model ./test_models/gru_model.py
lunchbox deploy --model test_models.cnn_model
```

## Final tasks

- implement train functionality, explore using mojo for optimizations
- implement deploy functionality by automatically generating a dockerfile
- test lastly

- error handling
- documentation of everything
- linting with ruff
- logo
- finalized readme
- make it public?
- license in repo?

## TODO LIST

- how can i integrate mojo project with this cli tool -- > utilize subprocess to run the mojo command

- ensure that all of the configuration options with simple python work!!

if inputted model file, i must be able to run the model: flag this stuff!!

- with just normal python
- with mojo
- other config
