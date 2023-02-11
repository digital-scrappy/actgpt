from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from hyperparameters import SPECIAL_TOKENS, VOCAB_SIZE
from paths import DATA_DIR, TOK_LOC


def main():
    tokenizer = Tokenizer(WordPiece())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    tokenizer.train([str(DATA_DIR / "train.txt")], trainer)
    tokenizer.save(str(TOK_LOC))


if __name__ == "__main__":
    main()
