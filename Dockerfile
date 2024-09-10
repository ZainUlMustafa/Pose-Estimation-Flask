FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
# RUN pip3 install Flask gunicorn flask_cors opencv-python imageio numpy mediapipe ultralytics flask-socketio pillow
RUN pip3 install Flask gunicorn flask_cors flask_socketio opencv-python-headless imageio numpy mediapipe ultralytics pillow

COPY src/ app/
WORKDIR /app

ENV PORT 8080
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
CMD exec python3 app.py