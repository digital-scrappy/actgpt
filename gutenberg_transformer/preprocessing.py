import random
from random import shuffle

import polars as pl

from hyperparameters import AUTHORS
from paths import DATA_DIR
from strip_headers import strip_headers


def main():
    random.seed(42)

    metadata = pl.read_csv(DATA_DIR / "metadata.csv")
    metadata = metadata.filter(pl.col("language") == "['de']")

    # train test split
    test_ids = []
    val_ids = []
    train_ids = []

    for author in AUTHORS:
        author_ids = metadata.filter((pl.col("author") == author))["id"].to_list()
        shuffle(author_ids)

        train_ids += author_ids[:-2]
        test_ids += [author_ids[-2]]
        val_ids += [author_ids[-1]]

    splits = {"train.txt": train_ids, "val.txt": val_ids, "test.txt": test_ids}

    for name, ids in splits.items():
        with open(DATA_DIR / name, "w") as out_f:
            for id in ids:
                try:
                    with open(DATA_DIR / "raw" / (id + "_raw.txt"), "r") as in_f:
                        raw_text = in_f.read()
                except FileNotFoundError:
                    continue
                text = strip_headers(raw_text)
                out_f.write("[SOS]")
                out_f.write(text)
                out_f.write("[EOS]")


if __name__ == "__main__":
    main()
