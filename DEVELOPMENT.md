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
lunchbox train --model test_models.lstm_model
lunchbox test --model test_models.gru_model
lunchbox deploy --model test_models.cnn_model
```

## Final tasks

- implement train functionality, explore using mojo for optimizations
- implement deploy functionality by automatically generating a dockerfile
- test lastly

- documentation of everything
- linting with ruff
- logo
- finalized readme
- license in repo?

## TODO LIST

- connect the files to allow for dev soon
- how can i integrate mojo project with this cli tool -- > utilize subprocess to run the mojo command
- setup test models
