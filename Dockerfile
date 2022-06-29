FROM python:3.8-slim-buster
ADD pyriemann_qiskit /pyriemann_qiskit
ADD examples /examples
ADD setup.py /
ADD README.md /

RUN apt-get update
RUN apt-get -y install git

RUN apt-get --allow-releaseinfo-change update
RUN python -m pip install --upgrade pip
RUN apt-get -y install --fix-missing git-core
RUN apt-get -y install build-essential

RUN python setup.py develop
RUN pip install .[docs]

# WORKAROUNDS
## Qiskit problem in parsing version number
RUN echo 0.21.0 > /usr/local/lib/python3.8/site-packages/qiskit_terra-0.21.0rc1-py3.8-linux-x86_64.egg/qiskit/VERSION.txt
## Creating folder for mne data
RUN mkdir /root/mne_data

ENTRYPOINT [ "python", "/examples/ERP/classify_P300_bi.py" ]