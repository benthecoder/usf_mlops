# MLOps Course Labs

This repository contains all the labs and projects for the MLOps course. It follows standardized development practices for machine learning operations.

## Environment Setup

This project uses a Python virtual environment with specific package requirements:

- mlflow==2.15.1
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1

### Setup Instructions

1. Clone this repository
2. Create a virtual environment:

   ```bash
   # Using uv
   uv venv
   source .venv/bin/activate

   # Using conda
   conda create --name mlops python=3.12
   conda activate mlops
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure

- `/notebooks`: Jupyter notebooks for explorations and analyses
- `/data`: Dataset storage
- `/models`: Trained models storage
- Additional folders will be added as needed during the course

## Course Progress

- Lab 1: Development Environment Setup ✅
