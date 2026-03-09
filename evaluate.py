# evaluate.py
# Evaluation script — loads a saved checkpoint and runs it on a test set.
#
# Outputs:
#   - UAS and LAS scores
#   - Per-sentence breakdown (optional)
#   - CoNLL-U file with predictions (optional, for official eval tools)
#
# Usage:
#   python evaluate.py \
#       --checkpoint checkpoints/best_model.pt \
#       --test_file  data/en_ewt-ud-test.conllu \
#       --vocab      checkpoints/vocab.pkl \
#       [--output    predictions.conllu] \
#       [--use_mst]

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from data.conllu  import read_conllu, write_conllu
from data.vocab   import Vocab
from data.dataset import build_dataloader
from utils.metric import AttachmentMetric
from utils.mst    import greedy_decode, batch_decode_mst


@torch.no_grad()
def predict(model, loader, vocab, device, use_mst: bool = True, is_bert: bool = False):
    model.eval()
    metric         = AttachmentMetric()
    all_pred_heads = []
    all_pred_rels  = []

    for batch in tqdm(loader, desc='Predicting', ncols=100):
        gold_heads = batch['heads'].to(device)
        gold_rels  = batch['rels'].to(device)
        mask       = batch['mask'].to(device)

        if is_bert:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_map       = batch['word_map'].to(device)
            arc_scores, label_scores = model(input_ids, attention_mask, word_map)
            lengths = mask.sum(dim=1).cpu()
        else:
            words  = batch['words'].to(device)
            tags   = batch['tags'].to(device)
            chars  = batch.get('chars')
            if chars is not None:
                chars = chars.to(device)
            arc_scores, label_scores = model(words, tags, chars)
            lengths = batch['lengths']

        if use_mst:
            pred_heads = batch_decode_mst(arc_scores, mask).to(device)
        else:
            pred_heads = greedy_decode(arc_scores, mask)

        pred_rels = model.predict_labels(label_scores, pred_heads)
        metric.update(pred_heads, pred_rels, gold_heads, gold_rels, mask)

        pred_heads_np = pred_heads.cpu().numpy()
        pred_rels_np  = pred_rels.cpu().numpy()
        lengths_np    = lengths.numpy()
        for i, length in enumerate(lengths_np):
            all_pred_heads.append(pred_heads_np[i, :length].tolist())
            all_pred_rels.append([vocab.id_to_rel(r) for r in pred_rels_np[i, :length]])

    return metric, all_pred_heads, all_pred_rels


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    vocab = Vocab.load(args.vocab)

    print(f"\nLoading test data: {args.test_file}")
    test_sents = read_conllu(args.test_file)
    print(f"  {len(test_sents)} sentences")

    # ---- Load checkpoint to detect model type ----
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config     = checkpoint['config']
    is_bert    = 'bert_model' in config

    # ---- Build data loader matching model type ----
    if is_bert:
        from model.bert_parser import BERTBiaffineParser, BERTDependencyDataset, bert_collate_fn
        from torch.utils.data  import DataLoader
        test_dataset = BERTDependencyDataset(test_sents, vocab, config['bert_model'])
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=bert_collate_fn)
        model = BERTBiaffineParser(
            n_rels        = vocab.n_rels,
            bert_model    = config['bert_model'],
            arc_mlp_dim   = config.get('arc_mlp_dim',   500),
            label_mlp_dim = config.get('label_mlp_dim', 100),
            mlp_dropout   = config.get('mlp_dropout',  0.33),
            bert_dropout  = config.get('bert_dropout',  0.1),
        ).to(device)
    else:
        from model.base_parser import BiaffineParser
        use_char = config.get('use_char', False)
        test_loader = build_dataloader(
            test_sents, vocab, batch_size=args.batch_size,
            shuffle=False, use_char=use_char,
        )
        model = BiaffineParser(
            n_words          = vocab.n_words,
            n_tags           = vocab.n_tags,
            n_rels           = vocab.n_rels,
            n_chars          = vocab.n_chars if use_char else 0,
            word_embed_dim   = config.get('word_embed_dim',   100),
            tag_embed_dim    = config.get('tag_embed_dim',    100),
            char_embed_dim   = config.get('char_embed_dim',    50),
            char_out_dim     = config.get('char_out_dim',     100),
            lstm_hidden      = config.get('lstm_hidden',      400),
            lstm_layers      = config.get('lstm_layers',        3),
            lstm_dropout     = config.get('lstm_dropout',    0.33),
            arc_mlp_dim      = config.get('arc_mlp_dim',      500),
            label_mlp_dim    = config.get('label_mlp_dim',    100),
            mlp_dropout      = config.get('mlp_dropout',     0.33),
            embed_dropout    = config.get('embed_dropout',   0.33),
            use_char         = use_char,
            use_pos          = config.get('use_pos', True),
        ).to(device)

    model.load_state_dict(checkpoint['model_state'])
    print(f"  Model type: {'BERT' if is_bert else 'BiLSTM'}")
    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')} "
          f"(best dev LAS: {checkpoint.get('best_las', '?'):.2f}%)")

    # ---- Run evaluation ----
    print(f"\nEvaluating (MST decoding: {args.use_mst})...")
    metric, pred_heads, pred_rels = predict(
        model, test_loader, vocab, device, use_mst=args.use_mst, is_bert=is_bert
    )

    # ---- Print results ----
    print(f"\n{'='*50}")
    print(f"  TEST RESULTS")
    print(f"{'='*50}")
    print(f"  UAS : {metric.UAS:.2f}%")
    print(f"  LAS : {metric.LAS:.2f}%")
    print(f"  Tokens evaluated: {metric.n_total:,}")
    print(f"{'='*50}\n")

    # ---- Write CoNLL-U output (optional) ----
    if args.output:
        write_conllu(test_sents, pred_heads, pred_rels, args.output)
        print(f"Predictions written to: {args.output}")

    return metric.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate biaffine dependency parser')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to saved model checkpoint (.pt)')
    parser.add_argument('--test_file',  type=str, required=True,
                        help='Path to test CoNLL-U file')
    parser.add_argument('--vocab',      type=str, required=True,
                        help='Path to vocab.pkl file')
    parser.add_argument('--output',     type=str, default=None,
                        help='Optional: write predictions to this CoNLL-U file')
    parser.add_argument('--use_mst',    action='store_true',
                        help='Use MST decoding (slower but ensures valid trees)')
    parser.add_argument('--use_char',   action='store_true',
                        help='Enable CharLSTM (must match training config)')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    evaluate(args)
