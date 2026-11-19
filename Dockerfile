# Self-contained model server. The Udacity simulator runs natively on the host
# (it's a 3D game and needs the GPU) and connects to this container on port 4567.
FROM python:3.11-slim-bookworm
# imgaug depends on full opencv-python, which needs these system libs
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY TestSimulation.py TrainingSimulation.py utlis.py recovery.py model.h5 ./
# Recovery agent's vision LLM: OpenRouter's free-model router. It only switches on
# when a key is passed at run time (-e LLM_API_KEY=...); never bake the key in here.
ENV LLM_BASE_URL=https://openrouter.ai/api/v1 \
    LLM_MODEL=openrouter/free
EXPOSE 4567
CMD ["python", "-u", "TestSimulation.py"]
