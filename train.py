# train.py
# Full training loop for the biaffine dependency parser.
#
# Supports:
#   - Baseline: BiLSTM + Biaffine (exact reproduction of Dozat & Manning 2017)
#   - + CharLSTM (Innovation 1)
#   - BERT + Biaffine (Innovation 2)
#
# Usage:
#   # Baseline
#   python train.py --config configs/base.yaml
#
#   # BERT
#   python train.py --config configs/bert.yaml
#
#   # With CharLSTM
#   python train.py --config configs/base.yaml --use_char

import os
import sys
import argparse
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from data.conllu  import read_conllu
from data.vocab   import Vocab
from data.dataset import build_dataloader
from utils.metric    import AttachmentMetric, LossMetric
from utils.mst       import greedy_decode, batch_decode_mst
from utils.scheduler import get_base_optimizer, AnnealingScheduler


# --------------------------------------------------------------------------- #
# Loss computation
# --------------------------------------------------------------------------- #
def compute_loss(arc_scores: torch.Tensor,
                 label_scores: torch.Tensor,
                 gold_heads: torch.Tensor,
                 gold_rels:  torch.Tensor,
                 mask:       torch.Tensor) -> torch.Tensor:
    """
    Computes cross-entropy loss for arc prediction and label prediction.

    Arc loss: predict the correct head for each dependent word.
    Label loss: given the gold head, predict the correct dependency relation.

    mask excludes padding AND the <root> token (index 0) from the loss.
    """
    # Build evaluation mask: real tokens, excluding root
    eval_mask = mask.clone()
    eval_mask[:, 0] = False   # root has no head to predict

    # --- Arc loss ---
    # arc_scores: [batch, dep, head]
    # Select only the real (non-pad, non-root) dependents
    arc_scores_flat  = arc_scores[eval_mask]          # [n_real, seq_len]
    gold_heads_flat  = gold_heads[eval_mask]           # [n_real]
    arc_loss = nn.CrossEntropyLoss()(arc_scores_flat, gold_heads_flat)

    # --- Label loss ---
    # label_scores: [batch, dep, head, n_rels]
    # First select real dependents, then gather the scores at the gold head
    label_scores_flat = label_scores[eval_mask]        # [n_real, seq_len, n_rels]
    # Gather at the gold head position
    head_idx = gold_heads_flat.view(-1, 1, 1).expand(
        -1, 1, label_scores_flat.size(-1)
    )
    scores_at_gold_head = label_scores_flat.gather(1, head_idx).squeeze(1)
    # scores_at_gold_head: [n_real, n_rels]
    gold_rels_flat = gold_rels[eval_mask]              # [n_real]
    label_loss = nn.CrossEntropyLoss()(scores_at_gold_head, gold_rels_flat)

    return arc_loss + label_loss


# --------------------------------------------------------------------------- #
# Batch forward pass — handles both BiLSTM and BERT batches
# --------------------------------------------------------------------------- #
def forward_batch(model, batch, device):
    """Routes a batch through the model regardless of whether it is a
    BiLSTM batch (has 'words'/'tags') or a BERT batch (has 'input_ids')."""
    if 'input_ids' in batch:
        # BERT batch
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        word_map       = batch['word_map'].to(device)
        return model(input_ids, attention_mask, word_map)
    else:
        # BiLSTM batch
        words = batch['words'].to(device)
        tags  = batch['tags'].to(device)
        chars = batch.get('chars')
        if chars is not None:
            chars = chars.to(device)
        return model(words, tags, chars)


# --------------------------------------------------------------------------- #
# One training epoch
# --------------------------------------------------------------------------- #
def train_epoch(model, loader, optimizer, scheduler, device, use_mst=False):
    model.train()
    loss_metric = LossMetric()
    metric      = AttachmentMetric()

    for batch in tqdm(loader, desc='  Train', ncols=100, leave=False):
        gold_heads = batch['heads'].to(device)
        gold_rels  = batch['rels'].to(device)
        mask       = batch['mask'].to(device)

        optimizer.zero_grad()

        arc_scores, label_scores = forward_batch(model, batch, device)

        loss = compute_loss(arc_scores, label_scores, gold_heads, gold_rels, mask)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        scheduler.step()

        loss_metric.update(loss.item())

        with torch.no_grad():
            pred_heads = greedy_decode(arc_scores, mask)
            pred_rels  = model.predict_labels(label_scores, pred_heads)
            metric.update(pred_heads, pred_rels, gold_heads, gold_rels, mask)

    return loss_metric.avg, metric


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, use_mst=True):
    model.eval()
    metric = AttachmentMetric()

    for batch in tqdm(loader, desc='  Eval ', ncols=100, leave=False):
        gold_heads = batch['heads'].to(device)
        gold_rels  = batch['rels'].to(device)
        mask       = batch['mask'].to(device)

        arc_scores, label_scores = forward_batch(model, batch, device)

        if use_mst:
            pred_heads = batch_decode_mst(arc_scores, mask).to(device)
        else:
            pred_heads = greedy_decode(arc_scores, mask)

        pred_rels = model.predict_labels(label_scores, pred_heads)
        metric.update(pred_heads, pred_rels, gold_heads, gold_rels, mask)

    return metric


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def train(config: dict):
    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ---- Load data ----
    print("\nLoading data...")
    train_sents = read_conllu(config['train_file'])
    dev_sents   = read_conllu(config['dev_file'])
    print(f"  Train: {len(train_sents)} sentences")
    print(f"  Dev:   {len(dev_sents)} sentences")

    # ---- Build vocabulary ----
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    vocab_path = os.path.join(save_dir, 'vocab.pkl')

    vocab = Vocab()
    vocab.build(train_sents, min_freq=config.get('min_freq', 2))
    vocab.save(vocab_path)

    # ---- Pre-trained embeddings (optional) ----
    pretrained = None
    if config.get('pretrained_embed'):
        pretrained = vocab.load_pretrained_embeds(
            config['pretrained_embed'],
            embed_dim=config.get('word_embed_dim', 100)
        )

    # ---- Detect model type from config ----
    use_bert   = 'bert_model' in config
    use_char   = config.get('use_char', False) and not use_bert
    batch_size = config.get('batch_size', 32)

    # ---- Data loaders ----
    if use_bert:
        from model.bert_parser import BERTDependencyDataset, bert_collate_fn
        from torch.utils.data import DataLoader
        bert_model_name = config['bert_model']
        train_dataset = BERTDependencyDataset(train_sents, vocab, bert_model_name)
        dev_dataset   = BERTDependencyDataset(dev_sents,   vocab, bert_model_name)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True,  collate_fn=bert_collate_fn)
        dev_loader    = DataLoader(dev_dataset,   batch_size=batch_size,
                                   shuffle=False, collate_fn=bert_collate_fn)
    else:
        train_loader = build_dataloader(train_sents, vocab, batch_size=batch_size,
                                        shuffle=True,  use_char=use_char)
        dev_loader   = build_dataloader(dev_sents,   vocab, batch_size=batch_size,
                                        shuffle=False, use_char=use_char)

    # ---- Build model ----
    print("\nBuilding model...")
    if use_bert:
        from model.bert_parser import BERTBiaffineParser
        from utils.scheduler   import get_bert_optimizer
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
            pretrained_embeds = pretrained,
            use_char         = use_char,
            use_pos          = config.get('use_pos', True),
        ).to(device)

    print(f"  Parameters: {model.count_parameters():,}")

    # ---- Optimizer + Scheduler ----
    if use_bert:
        optimizer = get_bert_optimizer(
            model,
            bert_lr = config.get('bert_lr', 2e-5),
            head_lr = config.get('head_lr', 1e-3),
        )
    else:
        optimizer = get_base_optimizer(
            model.parameters(), lr=config.get('lr', 2e-3)
        )
    scheduler = AnnealingScheduler(
        optimizer,
        decay_factor = config.get('decay_factor', 0.75),
        decay_steps  = config.get('decay_steps',  5000),
    )

    # ---- Training loop ----
    max_epochs  = config.get('max_epochs', 100)
    best_las    = 0.0
    best_epoch  = 0
    patience    = config.get('patience', 10)   # early stopping
    no_improve  = 0

    print(f"\nTraining for up to {max_epochs} epochs "
          f"(early stopping patience={patience})\n")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train UAS':>10} "
          f"{'Train LAS':>10} {'Dev UAS':>10} {'Dev LAS':>10}")
    print('-' * 65)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_metric = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )

        # Evaluate on dev set
        dev_metric = evaluate(model, dev_loader, device, use_mst=True)

        elapsed = time.time() - t0
        print(f"{epoch:>6d} {train_loss:>12.4f} {train_metric.UAS:>9.2f}% "
              f"{train_metric.LAS:>9.2f}% {dev_metric.UAS:>9.2f}% "
              f"{dev_metric.LAS:>9.2f}%  [{elapsed:.0f}s]")

        # Save best model
        if dev_metric.LAS > best_las:
            best_las   = dev_metric.LAS
            best_epoch = epoch
            no_improve = 0
            checkpoint = {
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict(),
                'config':      config,
                'best_las':    best_las,
            }
            torch.save(checkpoint,
                       os.path.join(save_dir, 'best_model.pt'))
            print(f"         *** New best LAS: {best_las:.2f}% — model saved ***")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\nTraining complete. Best Dev LAS: {best_las:.2f}% at epoch {best_epoch}")
    return best_las


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train biaffine dependency parser')
    parser.add_argument('--config',   type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--use_char', action='store_true',
                        help='Enable CharLSTM embeddings (Innovation 1)')
    parser.add_argument('--seed',     type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Command-line flags override config
    if args.use_char:
        config['use_char'] = True

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(config)
