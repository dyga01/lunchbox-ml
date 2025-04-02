# development notes

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
curl https://get.modular.com | sh
pip install -e .
```

```text
curl -ssL https://magic.modular.com/5cf2c264-40cf-42be-bf0f-aa05b5ab6fd1 | bash
export PATH=$HOME/.modular/bin:$PATH
echo 'export PATH=$HOME/.modular/bin:$PATH' >> ~/.zshrc
magic init life --format mojoproject
cd life
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
- logo and finalized readme.md documentation
- license in repo?

## TODO LIST

- connect the files to allow for dev soon
