```text
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── vocabulary.py       # Vocab builders, encoders, decoders
│   │   └── dataset.py          # Padding and batch stacking logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── layers.py           # Positional encodings, FFN, LayerNorm, Dropout
│   │   ├── attention.py        # Scaled dot-product and Multi-Head modules
│   │   ├── encoder.py          # EncoderLayer and stacked assembly
│   │   ├── decoder.py          # DecoderLayer and stacked assembly
│   │   └── transformer.py      # Full Seq2Seq wrapper, tying embeddings
│   │
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── adam.py             # Adam optimizer from scratch
│   │   └── scheduler.py        # Noam learning rate scheduling formulas
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loss.py             # Label-smoothed KL loss matrix ops
│   │   └── engine.py           # Init scripts, training step, loops, accuracy metrics
│   │
│   └── inference/
│       ├── __init__.py
│       └── beam_search.py      # Greedy decoding and length-penalized beam search
│
├── config.py                   # Model dimensions (d_model, heads, dropout rates)
├── train.py                    # Main script orchestration entry point
└── README.md                   # System description and performance baselines