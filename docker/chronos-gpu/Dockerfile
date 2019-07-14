FROM tensorflow/tensorflow:1.12.0-gpu-py3

USER root

RUN apt update && apt install -y git


RUN mkdir -p chronos \
&& mkdir -p chronos/src \
&& mkdir -p chronos/src/chronos \
&& mkdir -p chronos/templates \
&& mkdir -p chronos/advisor

WORKDIR /chronos

COPY advisor advisor
COPY src src
COPY templates templates
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -e src
RUN pip install -r requirements.txt
RUN pip install -r advisor/requirements.txt
RUN pip install -r advisor/advisor_client/requirements.txt
RUN pip install -e advisor/advisor_client/
