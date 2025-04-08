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
exit --> to exit the mojo shell
```

```text
lunchbox --help
lunchbox train --model ./test_models/gru_model.py
lunchbox train --model ./test_models/gru_model.py --output
lunchbox train --model ./test_models/gru_model.py --output --benchmark
lunchbox train --model ./test_models/gru_model.py --output --benchmark --optimize mojo
lunchbox train --model ./test_models/gru_model.py --optimize mojo
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

how can i integrate mojo to automatically run
`lunchbox train --model ./test_models/gru_model.py --output --benchmark --optimize mojo`
whenever i run this command. it should automatically run the main.mojo file by starting the magic shell 'magic shell' and running the command 'mojo run main.mojo'
MAKE A SETUP BASH SCRIPT AND TEST EASIER SETUP
x

gpu integration
mojo support
magic support
other languages
