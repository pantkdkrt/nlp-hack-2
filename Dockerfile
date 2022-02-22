FROM python:3.9

COPY . /app

WORKDIR /app
RUN python -m pip install --upgrade pip

RUN pip --no-cache-dir install -r requirements.txt



ENTRYPOINT ["python3"]

EXPOSE 5000

CMD ["app.py"]
