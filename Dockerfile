# FROM python:3.7
FROM python

# RUN pip install virtualenv
# ENV VIRTUAL_ENV=/venv
# RUN virtualenv venv -p python3
# ENV PATH="VIRTUAL_ENV/bin:$PATH"

# WORKDIR /app
# ADD . /app

# Make a directory for our application
WORKDIR /BikeShare

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy our source code 
COPY . .
# Expose port
ENV PORT 8080

# Run the application:
CMD ["python", "BikePred.py", "--config=config.py"]