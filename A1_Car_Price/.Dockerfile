# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY ./code /app

# Install required Python libraries
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install scikit-learn

# Expose a port if your app needs one (optional)
EXPOSE 6000

# Command to run your app
CMD ["python3", "app.py"]
