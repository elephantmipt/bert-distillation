FROM pytorch/pytorch:latest

WORKDIR /exp

RUN apt-get update -y

RUN apt-get install libglib2.0-0 -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y libsm6 libxext6 libxrender-dev wget gzip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN ls

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements/requirements.txt

RUN bash bin/download_lenta.sh

RUN python scripts/split_dataset.py --small --sample 10000

CMD ["catalyst-dl", "run", "-C", "configs/config_ru_ranger.yml", "--verbose", "--distributed"]
