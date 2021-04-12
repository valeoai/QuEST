FROM nvidia/cuda:10.0-devel-ubuntu18.04

#RUN yes | unminimize

RUN apt-get update && apt-get install -y wget bzip2
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes

RUN conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.0 -c pytorch
RUN pip install scikit-image tqdm pyyaml easydict future pip
RUN apt-get install unzip
RUN conda install faiss-cpu -c pytorch
RUN conda install faiss-gpu cudatoolkit=10.0 -c pytorch

COPY ./ /quest
RUN pip install -e /quest

WORKDIR /quest

# Test importsS
RUN python -c "import scripts.main_classification"
