FROM python

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app/src

RUN pip3 install numpy opencv-python

COPY . .

CMD ["python3", "main.py"]