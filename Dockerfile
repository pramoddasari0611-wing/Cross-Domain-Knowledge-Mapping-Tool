# 1. Base Image: Use the official Python base image
# This provides the operating system (Linux) and Python pre-installed.
FROM python:3.10-slim

# 6. Set the working directory
# All subsequent commands will run inside this folder.
WORKDIR /usr/src/app

# 3. Copy project files
# First, copy the requirements file needed for step 4
COPY requirements.txt ./

# 4. Install Python dependencies
# Use the system-wide package installer, pip.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Run spaCy model downloads
# Although spaCy is not in the requirements above, it is essential for the project.
# We will download a small model for demonstration.
RUN python -m spacy download en_core_web_sm

# 3. Copy remaining project files
# Copy the entire project content into the container (including ui/app.py and data/).
COPY . .

# 7. Expose Port
# Tell Docker that the container listens on port 8501 (Streamlit's default port).
EXPOSE 8501

# 8. Define the startup command
# This command is executed when the container starts.
# Note: Assuming your Streamlit app file is located at 'ui/app.py'

# Replace the previous CMD instruction (e.g., 'streamlit_app.py') with the correct quoted name:
CMD ["streamlit", "run", "Extract Relation.py", "--server.port=8501", "--server.enableCORS=true"]