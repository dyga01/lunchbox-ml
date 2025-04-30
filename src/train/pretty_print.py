"""Printing functions for model benchmarking results and running messages."""

import os
import time

def print_benchmark_results(model_name, metrics, show_output=True, show_error=True):
    """
    Print formatted model results with color coding.
    
    Args:
        model_name (str): Name of the model that was executed
        metrics (dict): Dictionary containing model metrics from run_model function (output, error, execution_time, return_code)
        show_output (bool): Whether to show the model's standard output
        show_error (bool): Whether to show error messages even if execution was successful
    """
    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
    
    # Helper function to strip ANSI codes for length calculation
    def strip_ansi(text):
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    # Box dimensions
    BOX_WIDTH = 48
    CONTENT_WIDTH = BOX_WIDTH - 2  # Account for left and right border characters
    
    # Header
    print(f"\n{BOLD}{CYAN}{'╔' + '═' * BOX_WIDTH + '╗'}{RESET}")
    
    # Title section with model name
    title = f"MODEL: {os.path.basename(model_name)}"
    title_padding = CONTENT_WIDTH - len(title)
    print(f"{BOLD}{CYAN}║{RESET} {BOLD}{title}{RESET}{' ' * title_padding} {BOLD}{CYAN}║{RESET}")
    
    # Success status
    success = metrics['return_code'] == 0
    status_color = GREEN if success else RED
    status_text = "SUCCESS" if success else "FAILED"
    
    # Execution metrics
    execution_time = metrics['execution_time']
    time_color = GREEN if execution_time < 10 else (YELLOW if execution_time < 30 else RED)

    # Print benchmarking metrics if available
    if "peak_memory" in metrics and "average_cpu" in metrics:
        # Add Benchmark title
        print(f"{BOLD}{CYAN}╠{'─' * BOX_WIDTH}╣{RESET}")
        benchmark_title = "BENCHMARKS:"
        benchmark_padding = CONTENT_WIDTH - len(benchmark_title)
        print(f"{BOLD}{CYAN}║{RESET} {BOLD}{MAGENTA}{benchmark_title}{RESET}{' ' * benchmark_padding} {BOLD}{CYAN}║{RESET}")

        # Fix consistent spacing for metrics
        memory_text = f"Peak Memory Usage: {metrics['peak_memory']:.2f} MB"
        memory_padding = CONTENT_WIDTH - len(memory_text)
        print(f"{BOLD}{CYAN}║{RESET} {memory_text}{' ' * memory_padding} {BOLD}{CYAN}║{RESET}")
        
        cpu_text = f"Average CPU Usage: {metrics['average_cpu']:.2f}%"
        cpu_padding = CONTENT_WIDTH - len(cpu_text)
        print(f"{BOLD}{CYAN}║{RESET} {cpu_text}{' ' * cpu_padding} {BOLD}{CYAN}║{RESET}")

        # Fixed time formatting
        time_str = f"{execution_time:.2f}"
        display_time = f"Execution time: {time_color}{time_str} seconds{RESET}"
        visual_length = len(f"Execution time: {time_str} seconds")
        time_padding = CONTENT_WIDTH - visual_length
        print(f"{BOLD}{CYAN}║{RESET} {display_time}{' ' * time_padding} {BOLD}{CYAN}║{RESET}")
        
        # Fixed return code formatting
        return_code_str = str(metrics['return_code'])
        display_rc = f"Return code: {status_color}{return_code_str}{RESET}"
        visual_length = len(f"Return code: {return_code_str}")
        rc_padding = CONTENT_WIDTH - visual_length
        print(f"{BOLD}{CYAN}║{RESET} {display_rc}{' ' * rc_padding} {BOLD}{CYAN}║{RESET}")

    
    # Print output if requested and available
    if show_output and metrics.get('output'):
        # Add Benchmark title
        print(f"{BOLD}{CYAN}╠{'─' * BOX_WIDTH}╣{RESET}")
        output_title = "OUTPUT:"
        output_padding = CONTENT_WIDTH - len(output_title)
        print(f"{BOLD}{CYAN}║{RESET} {BOLD}{MAGENTA}{output_title}{RESET}{' ' * output_padding} {BOLD}{CYAN}║{RESET}")
        
        # Split output lines and format them
        output_lines = metrics['output'].split('\n')
        for line in output_lines:
            # Truncate line if too long and add ellipsis
            display_line = line
            if len(line) > CONTENT_WIDTH - 2:
                display_line = line[:CONTENT_WIDTH - 5] + "..."
            line_padding = CONTENT_WIDTH - len(display_line)
            print(f"{BOLD}{CYAN}║{RESET} {display_line}{' ' * line_padding} {BOLD}{CYAN}║{RESET}")
    
    # Print errors if they exist and are requested
    if show_error and metrics.get('error'):
        print(f"{BOLD}{CYAN}╠{'─' * BOX_WIDTH}╣{RESET}")
        error_header = "ERRORS/WARNINGS:"
        error_padding = CONTENT_WIDTH - len(error_header)
        print(f"{BOLD}{CYAN}║{RESET} {BOLD}{RED}{error_header}{RESET}{' ' * error_padding} {BOLD}{CYAN}║{RESET}")
        print(f"{BOLD}{CYAN}╠{'─' * BOX_WIDTH}╣{RESET}")
        
        # Split error lines and format them with red color
        error_lines = metrics['error'].split('\n')
        for line in error_lines:
            # Truncate line if too long and add ellipsis
            display_line = line
            if len(line) > CONTENT_WIDTH - 2:
                display_line = line[:CONTENT_WIDTH - 5] + "..."
                
            # Calculate padding without ANSI codes
            colored_line = f"{RED}{display_line}{RESET}"
            line_padding = CONTENT_WIDTH - len(display_line)  # Use uncolored length for padding
            print(f"{BOLD}{CYAN}║{RESET} {colored_line}{' ' * line_padding} {BOLD}{CYAN}║{RESET}")
    
    # Summary
    print(f"{BOLD}{CYAN}╠{'─' * BOX_WIDTH}╣{RESET}")
    
    # Format status line consistently
    status_display = f"Status: {status_color}{BOLD}{status_text}{RESET}"
    visual_length = len(f"Status: {status_text}")
    
    if not success:
        code_text = f" (code {metrics['return_code']})"
        status_display += f"{status_color}{code_text}{RESET}"
        visual_length += len(code_text)
    
    status_padding = CONTENT_WIDTH - visual_length
    print(f"{BOLD}{CYAN}║{RESET} {status_display}{' ' * status_padding} {BOLD}{CYAN}║{RESET}")
    
    # Footer
    print(f"{BOLD}{CYAN}{'╚' + '═' * BOX_WIDTH + '╝'}{RESET}\n")

def print_running_message(model_path):
    """Display an attractive 'Running model' message with a small animation."""
    # ANSI color codes
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    # Animation frames
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    # Get just the filename for cleaner display
    model_name = os.path.basename(model_path)
    
    # Print initial message
    print(f"{BOLD}{CYAN}▶ Running model:{RESET} {MAGENTA}{model_name}{RESET}")
    
    # Show a brief animation (3 cycles)
    for _ in range(3):
        for frame in frames:
            print(f"\r{BOLD}{CYAN}  {frame} Initializing...{RESET}", end="", flush=True)
            time.sleep(0.05)
    
    # Clear the animation line
    print("\r" + " " * 40 + "\r", end="", flush=True)
