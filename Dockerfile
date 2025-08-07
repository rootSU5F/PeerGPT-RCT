# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files from current dir into container
COPY . .

# Upgrade pip and install all Python packages
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
