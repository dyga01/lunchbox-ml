# empty for now
'''          # Run the model using subprocess from the model's directory
        if optimize and optimize.lower() == "mojo":
            # Change directory to the optimizer folder
            optimizer_dir = os.path.join(os.path.dirname(__file__), "../../optimizer")
            process = subprocess.Popen(
                ["magic", "shell"],  # Start the magic shell
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=optimizer_dir
            )
            # Write the command to run the Mojo file within the magic shell
            process.stdin.write("mojo run main.mojo\n")
            process.stdin.flush()  # Ensure the command is sent
            process.stdin.close()  # Close stdin after writing the command
'''