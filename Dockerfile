FROM tensorflow/tensorflow:latest-gpu-py3
ADD . /app
RUN pip3 install -r /app/config/req.list
WORKDIR /app
ENTRYPOINT [ "python3", "app.py" ]