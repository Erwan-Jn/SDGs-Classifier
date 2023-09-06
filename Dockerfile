  FROM python:3.10.6-buster

  COPY model /model
  COPY scripts /scripts
  COPY sdg_classifier_api /sdg_classifier_api
  COPY requirements_docker.txt /requirements.txt

  RUN pip install --upgrade pip
  RUN pip install -r requirements.txt

  CMD uvicorn sdg_classifier_api.fast:app --host 0.0.0.0 --port $PORT
