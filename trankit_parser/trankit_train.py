# trankit_train.py
# Train Trankit's joint POS + Dependency Parsing model on UD English EWT.
#
# Uses TPipeline with task='posdep':
#   - Backbone : XLM-RoBERTa (downloaded from HuggingFace — no UOregon needed)
#   - Adapters : small Up/Down projection networks injected in every transformer
#                layer (XLM-RoBERTa weights stay FROZEN — only adapters train)
#   - Joint    : POS tagging + morphological tagging + dependency parsing in one model
#
# This is the exact Trankit architecture from the paper (EACL 2021).
#
# Usage:
#   cd /Users/skolgane/trankit_parser
#   python3 trankit_train.py

from trankit import TPipeline
import os

TRAIN_CONLLU = '../biaffine_parser/data/en_ewt-ud-train.conllu'
DEV_CONLLU   = '../biaffine_parser/data/en_ewt-ud-dev.conllu'
SAVE_DIR     = './checkpoints/trankit_posdep'

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("  TRANKIT — Joint POS + Dependency Parsing")
print("  Backbone : xlm-roberta-base (HuggingFace)")
print("  Method   : Adapters (XLM-RoBERTa frozen)")
print("  Data     : UD English EWT")
print("=" * 60)
print(f"\n  Train : {TRAIN_CONLLU}")
print(f"  Dev   : {DEV_CONLLU}")
print(f"  Save  : {SAVE_DIR}\n")

tp = TPipeline(training_config={
    'task':               'posdep',       # joint POS + dependency parsing
    'save_dir':           SAVE_DIR,
    'train_conllu_fpath': TRAIN_CONLLU,
    'dev_conllu_fpath':   DEV_CONLLU,
    'embedding':          'xlm-roberta-base',
    'category':           'customized',   # custom language label
    'gpu':                False,          # CPU-only
    'batch_size':         16,
    'max_epoch':          30,
})

print("\nStarting training...")
tp.train()
print(f"\nTraining complete. Model saved to: {SAVE_DIR}")
