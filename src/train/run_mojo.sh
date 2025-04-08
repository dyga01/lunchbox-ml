#!/bin/bash

# Navigate to the optimizer directory
cd optimizer || exit

# Use magic exec to run the Mojo command in the environment
magic exec mojo main.mojo
