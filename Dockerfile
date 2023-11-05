FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile","Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "./"]

EXPOSE 2222

ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:2222", "predict:app" ]