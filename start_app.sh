#!/bin/bash



# Start the Streamlit app
echo "Starting Streamlit app..."
PYTHONPATH=$(pwd) streamlit run src/ui/app.py
