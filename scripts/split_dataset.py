import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def main(args):
    """Splitting dataset"""
    df = pd.read_csv("data/lenta-ru-news.csv", usecols=["text"]).dropna()
    if args.sample is not None:
        print(f"sampling {args.sample} texts")
        df = df.sample(args.sample, random_state=args.sample)
    train, valid = train_test_split(
        df, random_state=args.random_state, test_size=2000
    )
    train = train.reset_index()
    valid = valid.reset_index()
    if args.small:
        train.to_csv("data/train_small.csv")
        valid.to_csv("data/valid_small.csv")
    else:
        train.to_csv("data/train.csv")
        valid.to_csv("data/valid.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", default=42, type=int, required=False)
    parser.add_argument("--sample", default=None, type=int, required=False)
    args = parser.parse_args()
    main(args)
