import re
import pandas as pd
filename = 'data/lenta-ru-news.csv'
df = pd.read_csv(filename)

batch_size = 512
batched_df = ['']
curlen = 0


def make_words(sentence):
    arr = re.findall('[а-яА-Я]+', sentence)
    return len(arr), ' '.join(arr)


for row in df.text:
    try:
        sentences = filter(lambda x: len(x) > 0, re.split('\\?|\\.|\\!', row))
    except:
        continue
    for sentence in sentences:
        count, sentence = make_words(sentence)
        if count > 512:
            continue

        if curlen + count > 512:
            batched_df.append('')
            curlen = 0
        batched_df[-1] += sentence + '.'
        curlen += count

batched_df = pd.DataFrame({'text': batched_df})
batched_df.to_csv(filename)
