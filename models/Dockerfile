FROM ubuntu

RUN apt-get update && \
  apt-get install -y \
    python-pip python-dev pandoc \
    libfreetype6-dev pkg-config \
    r-base-core libzmq3-dev libcurl4-openssl-dev \
    texlive-latex-base texlive-latex-extra texlive-fonts-recommended && \
  rm -rf /var/cache/apt

RUN pip install --upgrade pip

WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

COPY Lasagne-master.zip /usr/src/app/
RUN pip install --user --upgrade --no-deps Lasagne-master.zip

RUN pip install --upgrade --no-cache-dir https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl

EXPOSE 9123

CMD ["/bin/sh", "-c", "jupyter notebook --debug --notebook-dir=/usr/src/app/models --no-browser --port 9123 --ip=0.0.0.0"]
