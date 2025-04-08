#!/bin/bash

# Navigate to the optimizer directory
magic init optimizer --format mojoproject
cd optimizer

# Create the Mojo file with valid syntax
echo "# My first Mojo program!" > main.mojo
echo "def main():" >> main.mojo
echo "    print(\"Hello, World!\")" >> main.mojo

# Run the Mojo file
magic run mojo main.mojo
