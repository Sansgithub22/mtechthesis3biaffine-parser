# utils/mst.py
# Maximum Spanning Tree (MST) decoding for dependency parsing.
#
# At test time, the greedy prediction (each word picks its highest-scoring head)
# may produce a graph with cycles, which is not a valid dependency tree.
# The Chu-Liu / Edmonds algorithm finds the highest-scoring valid spanning
# arborescence (directed tree) from the arc score matrix.
#
# Algorithm reference:
#   - Chu & Liu (1965) "On the shortest arborescence of a directed graph"
#   - Edmonds (1967) "Optimum branchings"
#   - McDonald et al. (2005) "Non-projective Dependency Parsing using
#     Spanning Tree Algorithms" (the NLP adaptation we follow)
#
# We use networkx for the core MST computation, which implements Edmonds' algorithm.

import torch
import numpy as np
import networkx as nx
from typing import List


def decode_mst(arc_scores: np.ndarray) -> np.ndarray:
    """
    Finds the maximum spanning arborescence rooted at node 0 using
    Chu-Liu / Edmonds algorithm.

    Parameters
    ----------
    arc_scores : np.ndarray, shape [seq_len, seq_len]
        arc_scores[dep][head] = score of arc head → dep
        Index 0 is the root node.

    Returns
    -------
    heads : np.ndarray, shape [seq_len]
        heads[i] = predicted head of word i
        heads[0] = 0 (root points to itself by convention)
    """
    seq_len = arc_scores.shape[0]

    # Build a directed graph where each edge has a weight equal to the arc score
    # Edge direction: head → dep  (a word points toward its dependents)
    G = nx.DiGraph()
    G.add_nodes_from(range(seq_len))

    for dep in range(1, seq_len):       # word 0 is root, skip as dependent
        for head in range(seq_len):
            if head != dep:             # no self-loops
                G.add_edge(head, dep, weight=arc_scores[dep][head])

    # Chu-Liu / Edmonds maximum spanning arborescence rooted at node 0
    try:
        arborescence = nx.maximum_spanning_arborescence(G)
    except nx.exception.NetworkXException:
        # Fallback: greedy (shouldn't normally happen with complete graphs)
        return np.argmax(arc_scores, axis=1)

    heads = np.zeros(seq_len, dtype=np.int64)
    for head, dep in arborescence.edges():
        heads[dep] = head

    return heads


def batch_decode_mst(arc_scores: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """
    Applies MST decoding to each sentence in a batch.

    Parameters
    ----------
    arc_scores : torch.Tensor [batch, seq_len, seq_len]
        arc_scores[b, dep, head] = score of arc head→dep in sentence b
    mask : torch.Tensor [batch, seq_len]
        True for real tokens (including <root>)

    Returns
    -------
    pred_heads : torch.Tensor [batch, seq_len]
    """
    batch_size = arc_scores.size(0)
    pred_heads = torch.zeros_like(arc_scores[:, :, 0], dtype=torch.long)

    arc_scores_np = arc_scores.detach().cpu().numpy()
    mask_np       = mask.cpu().numpy()

    for b in range(batch_size):
        # Get the real length of sentence b (count True values in mask)
        length = int(mask_np[b].sum())

        # Extract the sub-matrix for real tokens only
        score_sub = arc_scores_np[b, :length, :length]

        # Run MST on this sentence
        heads = decode_mst(score_sub)

        pred_heads[b, :length] = torch.tensor(heads, dtype=torch.long)

    return pred_heads


def greedy_decode(arc_scores: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """
    Faster greedy decoding: each word picks its highest-scoring head.
    Used during training evaluation when speed matters more than tree validity.

    arc_scores : [batch, dep, head]
    mask :       [batch, seq_len]
    returns:     [batch, seq_len]
    """
    # Mask out padding positions so they are never selected as heads
    arc_scores = arc_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
    return arc_scores.argmax(dim=-1)  # [batch, seq_len]
