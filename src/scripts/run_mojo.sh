#!/bin/bash

# Navigate to the optimizer directory
magic init optimizer --format mojoproject
cd optimizer
magic run mojo main.mojo
