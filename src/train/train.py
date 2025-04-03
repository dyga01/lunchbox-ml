import subprocess
import time
import os

def run_model(model_path, report_output=False, benchmark=False):
    """
    Runs the model located at the given path using subprocess.

    Args:
        model_path (str): Path to the model file.
        report_output (bool): Whether to print the model's output.
        benchmark (bool): Whether to report performance benchmarks.

    Returns:
        dict: A dictionary containing the model output, execution time, and other metrics.
    """
    try:
        print(f"Starting model execution: {os.path.basename(model_path)}")
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
        
        if report_output:
            print(f"\n{'='*50}")
            print(f"MODEL OUTPUT:")
            print(f"{'='*50}")
            print(metrics["output"])
            if metrics["error"]:
                print(f"\nERRORS/WARNINGS:")
                print(metrics["error"])
            print(f"{'='*50}\n")
        
        # Always print a summary if either flag is set
        if benchmark or report_output:
            print(f"Model execution completed successfully in {execution_time:.2f} seconds")
        
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

def print_benchmark_results(model_name, metrics, show_output=True, show_error=True):
    """
    Print formatted model results.
    
    Args:
        model_name (str): Name of the model that was executed
        metrics (dict): Dictionary containing model metrics from run_model function (output, error, execution_time, return_code)
        show_output (bool): Whether to show the model's standard output
        show_error (bool): Whether to show error messages even if execution was successful
    """
    print(f"\n{'='*50}")
    print(f"MODEL RESULTS: {model_name}")
    print(f"{'='*50}")
    
    # Print execution metrics
    print(f"Execution time: {metrics['execution_time']:.2f} seconds")
    print(f"Return code: {metrics['return_code']}")
    
    # Print output if requested and available
    if show_output and metrics.get('output'):
        print(f"\nOUTPUT:")
        print(f"{'-'*30}")
        print(metrics['output'])
    
    # Print errors if they exist and are requested
    if show_error and metrics.get('error'):
        print(f"\nERRORS/WARNINGS:")
        print(f"{'-'*30}")
        print(metrics['error'])
    
    # Summary
    if metrics['return_code'] == 0:
        print(f"\nStatus: SUCCESS")
    else:
        print(f"\nStatus: FAILED (code {metrics['return_code']})")
    
    print(f"{'='*50}")
