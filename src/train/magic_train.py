'''
run the magic shell from the optimizer directory and run python code

cd optimizer
magic shell
mojo main.mojo
'''
import subprocess

def run_magic():
    subprocess.run(["bash", "src/scripts/run_magic.sh"], check=True)
