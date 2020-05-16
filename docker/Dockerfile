FROM pytorch/pytorch:latest

WORKDIR /exp

RUN apt-get update -y

RUN apt install libglib2.0-0 -y

RUN apt install -y libsm6 libxext6 libxrender-dev wget gzip

COPY . .

RUN ls

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements/requirements.txt

RUN bash bin/download_lenta.sh

RUN python scripts/split_dataset.py --small --sample 10000

CMD ["catalyst-dl", "run", "-C", "configs/config_ru_ranger.yml", "--verbose", "--distributed"]
