# Self-contained model server. The Udacity simulator runs natively on the host
# (it's a 3D game and needs the GPU) and connects to this container on port 4567.
FROM python:3.11-slim-bookworm
# imgaug depends on full opencv-python, which needs these system libs
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY TestSimulation.py TrainingSimulation.py utlis.py model.h5 ./
EXPOSE 4567
CMD ["python", "-u", "TestSimulation.py"]
