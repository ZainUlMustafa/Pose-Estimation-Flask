FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install Flask gunicorn flask_cors opencv-python imageio numpy mediapipe ultralytics

COPY src/ app/
WORKDIR /app

ENV PORT 8080
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
CMD exec python3 app.py