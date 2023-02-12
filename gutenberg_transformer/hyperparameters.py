SPECIAL_TOKENS = ["[UNK]", "[SOS]", "[EOS]"]
AUTHORS = [
    "Goethe, Johann Wolfgang von",
    "Schiller, Friedrich",
    "Raimund, Ferdinand",
    "Lessing, Gotthold Ephraim",
]

# AUTHORS = ["Goethe, Johann Wolfgang von", "Schiller, Friedrich", "Raimund, Ferdinand", "Lessing, Gotthold Ephraim", "Büchner, Georg"]
VOCAB_SIZE = 512
BLOCK_SIZE = 10
BATCH_SIZE = 32  # how many independent sequences will we process in parallel?
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
EVAL_ITERS = 200
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
