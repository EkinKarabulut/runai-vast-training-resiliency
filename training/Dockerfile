FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /demo

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt distributed.py launch.sh /demo/

RUN pip install --no-cache-dir -r requirements.txt \
    && chmod +x launch.sh

CMD ["./launch.sh"]
