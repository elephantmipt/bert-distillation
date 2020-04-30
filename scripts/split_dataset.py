import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def main(args):
    """Splitting dataset"""
    df = pd.read_csv("data/lenta-ru-news.csv")
    if args.small:
        print("sampling 15000 texts")
        df = df.sample(15000, random_state=args.random_state)
    train, valid = train_test_split(df, random_state=args.random_state)
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
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--random_state", default=42, type=int, required=False)
    args = parser.parse_args()
    main(args)
