# trankit_evaluate.py
# Evaluate the trained Trankit posdep model on UD English EWT test set.
#
# Uses TPipeline internals directly — no full Pipeline needed (avoids requiring
# tokenizer.mdl which is not produced by task='posdep' training).
#
# Strategy:
#   1. Instantiate TPipeline with test file as dev (for vocab + dataset setup)
#   2. Load saved adapter weights from customized.tagger.mdl
#   3. Call TPipeline._eval_posdep on the test set
#   4. Read the conllu output and compute UAS / LAS
#
# Usage:
#   cd /Users/skolgane/trankit_parser
#   python3 trankit_evaluate.py \
#       --save_dir   checkpoints/trankit_posdep \
#       --train_file ../biaffine_parser/data/en_ewt-ud-train.conllu \
#       --test_file  ../biaffine_parser/data/en_ewt-ud-test.conllu

import argparse
import os
import torch
from trankit import TPipeline


# --------------------------------------------------------------------------- #
# UAS / LAS from CoNLL-U files
# --------------------------------------------------------------------------- #
def score_conllu(pred_path, gold_path):
    """Compute UAS and LAS by comparing two CoNLL-U files token by token."""
    def read(path):
        sents, cur = [], []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('#') or line == '':
                    if line == '' and cur:
                        sents.append(cur); cur = []
                    continue
                fields = line.split('\t')
                if '-' in fields[0] or '.' in fields[0]:
                    continue
                cur.append((fields[6], fields[7]))   # (head, deprel)
        if cur:
            sents.append(cur)
        return sents

    pred_sents = read(pred_path)
    gold_sents = read(gold_path)
    n_total = n_arc = n_las = 0
    for ps, gs in zip(pred_sents, gold_sents):
        for (ph, pr), (gh, gr) in zip(ps, gs):
            n_total += 1
            if ph == gh:
                n_arc += 1
                if pr.lower() == gr.lower():
                    n_las += 1
    uas = 100.0 * n_arc / n_total if n_total else 0.0
    las = 100.0 * n_las / n_total if n_total else 0.0
    return uas, las, n_total


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #
def evaluate(args):
    mdl_path = os.path.join(args.save_dir, 'xlm-roberta-base', 'customized',
                            'customized.tagger.mdl')
    assert os.path.exists(mdl_path), f"Model not found: {mdl_path}"
    print(f"\nModel checkpoint : {mdl_path}")

    # ---- Build TPipeline (rebuilds vocab from train, sets up test as 'dev') ----
    print("Setting up TPipeline (this loads XLM-RoBERTa — may take ~1 min)...")
    tp = TPipeline(training_config={
        'task':               'posdep',
        'save_dir':           args.save_dir,
        'train_conllu_fpath': args.train_file,
        'dev_conllu_fpath':   args.test_file,   # use test as dev for eval
        'embedding':          'xlm-roberta-base',
        'category':           'customized',
        'gpu':                False,
        'batch_size':         16,
        'max_epoch':          0,               # no training
    })

    # ---- Load saved adapter + task-head weights ----
    print(f"Loading saved weights from: {mdl_path}")
    state = torch.load(mdl_path, map_location='cpu')
    adapters = state['adapters']
    epoch    = state.get('epoch', '?')
    print(f"  Checkpoint epoch: {epoch}")

    # Restore to embedding layers
    emb_sd = tp._embedding_layers.state_dict()
    for k, v in adapters.items():
        if k in emb_sd:
            emb_sd[k] = v
    tp._embedding_layers.load_state_dict(emb_sd)

    # Restore to tagger
    tag_sd = tp._tagger.state_dict()
    for k, v in adapters.items():
        if k in tag_sd:
            tag_sd[k] = v
    tp._tagger.load_state_dict(tag_sd)

    # ---- Evaluate on test set (stored as tp.dev_set) ----
    print(f"\nEvaluating on test set ({len(tp.dev_set)} sentences)...")
    score, pred_path = tp._eval_posdep(
        data_set  = tp.dev_set,
        batch_num = tp.dev_batch_num,
        name      = 'test',
        epoch     = epoch,
    )

    # ---- Read UAS / LAS from the written conllu ----
    uas, las, n_toks = score_conllu(pred_path, args.test_file)

    print(f"\n{'='*60}")
    print(f"  TRANKIT (XLM-RoBERTa + Adapters) — TEST RESULTS")
    print(f"  Joint POS + Dependency Parsing | EACL 2021")
    print(f"{'='*60}")
    print(f"  UAS : {uas:.2f}%")
    print(f"  LAS : {las:.2f}%")
    print(f"  Tokens evaluated: {n_toks:,}")
    print(f"  (checkpoint epoch {epoch})")
    print(f"{'='*60}")
    print(f"\n  Published Trankit (Table 1, EACL 2021):")
    print(f"  UAS: 90.14%  LAS: 87.96%  (full pipeline, raw text)")
    print(f"  Note: our eval uses gold tokenisation → expect higher numbers\n")

    if args.output and pred_path != args.output:
        import shutil
        shutil.copy(pred_path, args.output)
        print(f"Predictions written to: {args.output}")

    return uas, las


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained Trankit posdep model on CoNLL-U test set')
    parser.add_argument('--save_dir',   type=str,
                        default='checkpoints/trankit_posdep')
    parser.add_argument('--train_file', type=str,
                        default='../biaffine_parser/data/en_ewt-ud-train.conllu')
    parser.add_argument('--test_file',  type=str,
                        default='../biaffine_parser/data/en_ewt-ud-test.conllu')
    parser.add_argument('--output',     type=str, default=None)
    args = parser.parse_args()
    evaluate(args)
