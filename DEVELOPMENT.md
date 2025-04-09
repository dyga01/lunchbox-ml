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
```

```text
lunchbox train --model ./test_models/gru_model.py --optimize magic
lunchbox train --model ./test_models/gru_model.py --optimize max
```

## Final tasks

- implement train functionality, explore using mojo for optimizations
- implement deploy functionality by automatically generating a dockerfile?
- test lastly

- make the tool useable, this may include a lot of code that will automatically install models dependencies and logic around this
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
gpu integration, mojo support, other languages
how can i access my gpu to speed up the training of gru_model.py
[https://docs.modular.com/mojo/manual/gpu/intro-tutorial]
could i optimize my computers gpu and cpu with mojo and then run the python program with magic? how would i connect to these optimizations?
can i use mojo to optimize gpu kernel
and then use python to run on that optimized kernel
x

magic to compile hybrid python mojo approach
