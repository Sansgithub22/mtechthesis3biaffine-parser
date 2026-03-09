# eval_ensemble.py
# Innovation D: BERT + XLM-RoBERTa Ensemble
#
# Averages arc score matrices and label score matrices from two independently
# trained transformer-based parsers before MST decoding.
#
#   arc_scores   = w1 * arc_bert   + w2 * arc_xlmr
#   label_scores = w1 * label_bert + w2 * label_xlmr
#
# Because BERT and XLM-RoBERTa have different tokenizers, two separate
# DataLoaders are built and iterated in sync (both use shuffle=False and
# the same sentence order, so zip() is safe).
#
# Usage:
#   python eval_ensemble.py \
#       --ckpt_a   checkpoints/bert/best_model.pt \
#       --ckpt_b   checkpoints/xlmr/best_model.pt \
#       --vocab    checkpoints/bert/vocab.pkl \
#       --test_file data/en_ewt-ud-test.conllu \
#       [--weight_a 0.5] [--weight_b 0.5] \
#       [--output  predictions_ensemble.conllu] \
#       [--use_mst]

import argparse
import torch
import numpy as np
from tqdm import tqdm

from data.conllu  import read_conllu, write_conllu
from data.vocab   import Vocab
from utils.metric import AttachmentMetric
from utils.mst    import greedy_decode, batch_decode_mst
from model.bert_parser import (
    BERTBiaffineParser, BERTDependencyDataset, bert_collate_fn
)
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Model loader — handles any BERTBiaffineParser checkpoint
# --------------------------------------------------------------------------- #
def load_bert_model(ckpt_path: str, vocab, device):
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    assert 'bert_model' in config, \
        f"{ckpt_path} is not a BERT/XLM-R checkpoint (no 'bert_model' key in config)"

    model = BERTBiaffineParser(
        n_rels        = vocab.n_rels,
        bert_model    = config['bert_model'],
        arc_mlp_dim   = config.get('arc_mlp_dim',   500),
        label_mlp_dim = config.get('label_mlp_dim', 100),
        mlp_dropout   = config.get('mlp_dropout',  0.33),
        bert_dropout  = config.get('bert_dropout',  0.1),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    bert_name = config['bert_model']
    epoch     = ckpt.get('epoch', '?')
    best_las  = ckpt.get('best_las', float('nan'))
    print(f"  Loaded: {bert_name}  |  epoch {epoch}  |  dev LAS {best_las:.2f}%")
    return model, config


# --------------------------------------------------------------------------- #
# Build DataLoader for a specific BERT/XLM-R tokenizer
# --------------------------------------------------------------------------- #
def build_bert_loader(sents, vocab, bert_model_name: str, batch_size: int):
    dataset = BERTDependencyDataset(sents, vocab, bert_model_name)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,          # MUST be False — both loaders share same order
        collate_fn  = bert_collate_fn,
    )


# --------------------------------------------------------------------------- #
# Ensemble prediction loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_ensemble(
    model_a, loader_a,
    model_b, loader_b,
    vocab, device,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    use_mst: bool   = True,
):
    metric          = AttachmentMetric()
    all_pred_heads  = []
    all_pred_rels   = []

    loop = tqdm(
        zip(loader_a, loader_b),
        total   = len(loader_a),
        desc    = 'Ensemble predict',
        ncols   = 100,
    )

    for batch_a, batch_b in loop:
        # ---- shared supervision tensors (from either batch, same sentences) ----
        gold_heads = batch_a['heads'].to(device)
        gold_rels  = batch_a['rels'].to(device)
        mask       = batch_a['mask'].to(device)
        lengths    = mask.sum(dim=1).cpu()

        # ---- Model A forward (BERT) ----
        arc_a, lbl_a = model_a(
            batch_a['input_ids'].to(device),
            batch_a['attention_mask'].to(device),
            batch_a['word_map'].to(device),
        )

        # ---- Model B forward (XLM-RoBERTa) ----
        arc_b, lbl_b = model_b(
            batch_b['input_ids'].to(device),
            batch_b['attention_mask'].to(device),
            batch_b['word_map'].to(device),
        )

        # ---- Ensemble: weighted average of score matrices ----
        arc_scores   = weight_a * arc_a   + weight_b * arc_b    # [B, dep, head]
        label_scores = weight_a * lbl_a   + weight_b * lbl_b    # [B, dep, head, n_rels]

        # ---- Decode using combined scores ----
        if use_mst:
            pred_heads = batch_decode_mst(arc_scores, mask).to(device)
        else:
            pred_heads = greedy_decode(arc_scores, mask)

        # Use model_a's predict_labels (same label space, same MLP structure)
        pred_rels = model_a.predict_labels(label_scores, pred_heads)

        metric.update(pred_heads, pred_rels, gold_heads, gold_rels, mask)

        # Collect predictions for CoNLL-U output
        pred_heads_np = pred_heads.cpu().numpy()
        pred_rels_np  = pred_rels.cpu().numpy()
        lengths_np    = lengths.numpy()
        for i, length in enumerate(lengths_np):
            all_pred_heads.append(pred_heads_np[i, :length].tolist())
            all_pred_rels.append(
                [vocab.id_to_rel(r) for r in pred_rels_np[i, :length]]
            )

    return metric, all_pred_heads, all_pred_rels


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Ensemble weights:  model_a={args.weight_a:.2f}  model_b={args.weight_b:.2f}")

    # ---- Vocab (shared — both models trained on same treebank) ----
    vocab = Vocab.load(args.vocab)

    # ---- Test sentences ----
    print(f"\nLoading test data: {args.test_file}")
    test_sents = read_conllu(args.test_file)
    print(f"  {len(test_sents)} sentences")

    # ---- Load both models ----
    print(f"\nLoading model A: {args.ckpt_a}")
    model_a, config_a = load_bert_model(args.ckpt_a, vocab, device)

    print(f"\nLoading model B: {args.ckpt_b}")
    model_b, config_b = load_bert_model(args.ckpt_b, vocab, device)

    # ---- Build separate DataLoaders (different tokenizers) ----
    print("\nBuilding DataLoaders...")
    loader_a = build_bert_loader(
        test_sents, vocab, config_a['bert_model'], args.batch_size
    )
    loader_b = build_bert_loader(
        test_sents, vocab, config_b['bert_model'], args.batch_size
    )
    print(f"  Model A tokenizer : {config_a['bert_model']}")
    print(f"  Model B tokenizer : {config_b['bert_model']}")

    # ---- Run ensemble ----
    print(f"\nRunning ensemble (MST decoding: {args.use_mst})...")
    metric, pred_heads, pred_rels = predict_ensemble(
        model_a, loader_a,
        model_b, loader_b,
        vocab, device,
        weight_a = args.weight_a,
        weight_b = args.weight_b,
        use_mst  = args.use_mst,
    )

    # ---- Individual model scores for comparison ----
    print(f"\n{'='*58}")
    print(f"  INNOVATION D — BERT + XLM-RoBERTa ENSEMBLE RESULTS")
    print(f"{'='*58}")
    print(f"  Model A ({config_a['bert_model']:>20}) :")
    print(f"    Dev LAS trained: {torch.load(args.ckpt_a, map_location='cpu', weights_only=False).get('best_las', 0):.2f}%")
    print(f"  Model B ({config_b['bert_model']:>20}) :")
    print(f"    Dev LAS trained: {torch.load(args.ckpt_b, map_location='cpu', weights_only=False).get('best_las', 0):.2f}%")
    print(f"  {'─'*54}")
    print(f"  ENSEMBLE  UAS : {metric.UAS:.2f}%")
    print(f"  ENSEMBLE  LAS : {metric.LAS:.2f}%")
    print(f"  Tokens evaluated: {metric.n_total:,}")
    print(f"{'='*58}\n")

    # ---- Optional CoNLL-U output ----
    if args.output:
        write_conllu(test_sents, pred_heads, pred_rels, args.output)
        print(f"Predictions written to: {args.output}")

    return metric.UAS, metric.LAS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Innovation D: BERT + XLM-RoBERTa Ensemble'
    )
    parser.add_argument('--ckpt_a', type=str,
                        default='checkpoints/bert/best_model.pt',
                        help='Checkpoint for model A (e.g. BERT)')
    parser.add_argument('--ckpt_b', type=str,
                        default='checkpoints/xlmr/best_model.pt',
                        help='Checkpoint for model B (e.g. XLM-RoBERTa)')
    parser.add_argument('--vocab', type=str,
                        default='checkpoints/bert/vocab.pkl',
                        help='Shared vocab.pkl (same for both models)')
    parser.add_argument('--test_file', type=str,
                        default='data/en_ewt-ud-test.conllu',
                        help='Path to test CoNLL-U file')
    parser.add_argument('--weight_a', type=float, default=0.5,
                        help='Weight for model A arc/label scores (default 0.5)')
    parser.add_argument('--weight_b', type=float, default=0.5,
                        help='Weight for model B arc/label scores (default 0.5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: write ensemble predictions to CoNLL-U file')
    parser.add_argument('--use_mst', action='store_true',
                        help='Use MST decoding on combined scores (recommended)')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    evaluate(args)
