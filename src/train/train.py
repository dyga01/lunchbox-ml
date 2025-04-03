import subprocess
import time
import os
from .pretty_print import print_running_message

def run_model(model_path, output=False, benchmark=False, optimize=False):
    """
    Runs the model located at the given path using subprocess.

    Args:
        model_path (str): Path to the model file.
        output (bool): Whether to print the model's output.
        benchmark (bool): Whether to report performance benchmarks.

    Returns:
        dict: A dictionary containing the model output, execution time, and other metrics.
    """
    try:
        print_running_message(model_path)
    
        start_time = time.time()

        # Get the directory containing the model file
        model_dir = os.path.dirname(os.path.abspath(model_path))
        model_file = os.path.basename(model_path)

        # Run the model using subprocess from the model's directory
        result = subprocess.run(
            ["python", model_file],
            capture_output=True,
            text=True,
            check=True,
            cwd=model_dir  # This sets the working directory for the subprocess
        )
        
        end_time = time.time()
        execution_time = end_time - start_time

        # Collect metrics
        metrics = {
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
            "execution_time": execution_time,
            "return_code": result.returncode
        }
        
        # Report results based on flags
        if benchmark:
            print(f"\n{'='*50}")
            print(f"BENCHMARK RESULTS:")
            print(f"{'='*50}")
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Return code: {result.returncode}")
            print(f"{'='*50}\n")
        
        if output:
            print(f"\n{'='*50}")
            print(f"MODEL OUTPUT:")
            print(f"{'='*50}")
            print(metrics["output"])
            if metrics["error"]:
                print(f"\nERRORS/WARNINGS:")
                print(metrics["error"])
            print(f"{'='*50}\n")
        
        return metrics

    except subprocess.CalledProcessError as e:
        error_info = {
            "output": e.stdout.strip() if hasattr(e, 'stdout') else "",
            "error": f"An error occurred while running the model: {e}\n{e.stderr.strip() if hasattr(e, 'stderr') else ''}",
            "execution_time": time.time() - start_time,
            "return_code": e.returncode
        }
        
        print(f"\n{'='*50}")
        print(f"ERROR RUNNING MODEL:")
        print(f"{'='*50}")
        print(error_info["error"])
        print(f"{'='*50}\n")
        
        return error_info
