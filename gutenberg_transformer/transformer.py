import torch
from tokenizers import Tokenizer

from hyperparameters import BLOCK_SIZE
from paths import DATA_DIR, TOK_LOC


def build_ngrams(token_ids):
    X, Y = [], []
    
    # "[SOS]"  should have id 0
    # I am 99% the following line is wrong,
    # most probably this will cause the model to always predict the same start sequence
    # I'll look into splitting the text into paragraphs so I will have more starts like this
    
    context = [0 for _ in range(BLOCK_SIZE)]

    for token_id in token_ids:
        X.append(context)
        Y.append(token_id)

        context = context[1:] + [token_id]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


tokenizer = Tokenizer.from_file(str(TOK_LOC))
with open(DATA_DIR / "train.txt", "r") as f:
    train_enc = tokenizer.encode(f.read())
with open(DATA_DIR / "test.txt", "r") as f:
    test_enc = tokenizer.encode(f.read())

X, Y = build_ngrams(train_enc.ids)
