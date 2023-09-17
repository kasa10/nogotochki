FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && apt clean -y

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir albumentations opencv-python numpy streamlit

WORKDIR /app

COPY . /app/

EXPOSE 80

RUN ls -la

CMD [ "bash", "./run.sh" ]
