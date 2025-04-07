import os
import subprocess

def run_mojo(optimize=False):
    if optimize and optimize.lower() == "mojo":
        # Get optimizer directory path
        optimizer_dir = os.path.join(os.path.dirname(__file__), "../../optimizer")
        
        try:
            # Run magic shell and mojo command in one subprocess
            result = subprocess.run(
                "cd \"" + optimizer_dir + "\" && magic shell -c \"mojo main.mojo\"",
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Print output
            print(result.stdout)
            
            return result.stdout
            
        except subprocess.SubprocessError as e:
            print(f"Error running Mojo optimizer: {e}")
            return None
    
    return None