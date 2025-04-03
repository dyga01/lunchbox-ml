import subprocess
import time

def run_model(model_path):
    """
    Runs the model located at the given path using subprocess.

    Args:
        model_path (str): Path to the model file.

    Returns:
        dict: A dictionary containing the model output, execution time, and other metrics.
    """
    print("i am here")
    try:
        start_time = time.time()
        
        # Run the model using subprocess
        result = subprocess.run(
            ["python", model_path],
            capture_output=True,
            text=True,
            check=True
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

        return metrics

    except subprocess.CalledProcessError as e:
        return {
            "error": f"An error occurred while running the model: {e}",
            "return_code": e.returncode
        }
