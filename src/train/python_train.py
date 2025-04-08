import subprocess
import time
import os
import psutil
from .pretty_print import print_running_message, print_benchmark_results
from .mojo_train import run_mojo

def run_model(model_path, output=False, benchmark=False, optimize=False):
    """
    Runs the model located at the given path using subprocess.

    Args:
        model_path (str): Path to the model file.
        output (bool): Whether to print the model's output.
        benchmark (bool): Whether to report performance benchmarks.
        optimize (bool): Whether to run Mojo optimization.

    Returns:
        dict: A dictionary containing the model output, execution time, and other metrics.
    """
    try:
        print_running_message(model_path)
    
        start_time = time.time()

        # Initialize benchmarking variables
        peak_memory = 0
        cpu_usage = []

        # Get the directory containing the model file
        model_dir = os.path.dirname(os.path.abspath(model_path))
        model_file = os.path.basename(model_path)

        # Run the model using subprocess from the model's directory
        if optimize and optimize.lower() == "mojo":
            run_mojo()
            end_time = time.time()
            execution_time = end_time - start_time

            # Collect optimization metrics
            metrics = {
                "optimization": "mojo",
                "execution_time": execution_time,
                "status": "Optimization completed successfully",
                "return_code": 0  # Add a default return code for successful optimization
            }
            print_benchmark_results(model_file, metrics, show_output=output, show_error=True)
            return metrics
        else:
            process = subprocess.Popen(
                ["python", model_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=model_dir
            )

            # Monitor resource usage if benchmarking is enabled
            if benchmark:
                while process.poll() is None:
                    try:
                        # Record CPU usage
                        cpu_usage.append(psutil.cpu_percent(interval=0.1))
                        # Record peak memory usage
                        proc = psutil.Process(process.pid)
                        peak_memory = max(peak_memory, proc.memory_info().rss / (1024 * 1024))  # Convert to MB
                    except psutil.ZombieProcess:
                        # Handle zombie process gracefully
                        break
                    except psutil.NoSuchProcess:
                        # Process no longer exists
                        break
                    except psutil.AccessDenied:
                        # Access denied to process info
                        break

            stdout, stderr = process.communicate()
            end_time = time.time()
            execution_time = end_time - start_time

            # Collect metrics
            metrics = {
                "output": stdout.strip(),
                "error": stderr.strip(),
                "execution_time": execution_time,
                "return_code": process.returncode,
            }

            # Add benchmarking metrics if enabled
            if benchmark:
                metrics["peak_memory"] = peak_memory
                metrics["average_cpu"] = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0

            print_benchmark_results(model_file, metrics, show_output=output, show_error=True)
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
