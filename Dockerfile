FROM python:3.8-slim

# Image descriptions
LABEL maintainer="Mirko D'Andrea"
LABEL version="1.0"
LABEL description="Docker image for PROPAGATOR model"

RUN apt-get update && apt-get install -y gdal-bin libgdal-dev
RUN pip3 install pyarmor

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY build.sh .
ENTRYPOINT ["/bin/bash"]



