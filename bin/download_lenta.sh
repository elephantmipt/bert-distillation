mkdir -p data
wget -O ./data/lenta-ru-news.csv.gz https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz
gunzip -d data/lenta-ru-news.csv.gz
