'''
run the magic shell from the optimizer directory and run the mojo command

cd optimizer
magic shell
mojo main.mojo
'''
import subprocess

def run_mojo():
    subprocess.run(["bash", "src/train/scripts/run_mojo.sh"], check=True)
    # run the gru_model.py somehow with the mojo's magic compiler