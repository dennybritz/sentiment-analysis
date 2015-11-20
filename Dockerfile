FROM b.gcr.io/tensorflow/tensorflow

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install numpy scikit-learn pandas

COPY . /usr/src/app