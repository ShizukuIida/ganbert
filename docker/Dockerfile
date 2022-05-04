FROM sonoisa/deep-learning-coding:pytorch1.6.0_tensorflow2.3.0

WORKDIR /workspace
COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt

USER root

EXPOSE 8888