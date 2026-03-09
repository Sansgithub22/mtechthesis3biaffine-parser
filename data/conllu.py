# data/conllu.py
# Reads CoNLL-U formatted treebank files.
# CoNLL-U format has 10 tab-separated fields per token:
#   ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC

from dataclasses import dataclass, field
from typing import List


@dataclass
class Sentence:
    """Holds all fields for a single sentence."""
    words:   List[str]   # word forms,   index 0 = <root>
    lemmas:  List[str]   # lemma forms
    upos:    List[str]   # universal POS tags
    xpos:    List[str]   # language-specific POS tags
    heads:   List[int]   # gold head indices (0 = root)
    deprels: List[str]   # dependency relation labels
    chars:   List[List[str]] = field(default_factory=list)  # characters per word

    def __len__(self):
        return len(self.words)  # includes the synthetic <root> token


def read_conllu(path: str) -> List[Sentence]:
    """
    Parses a CoNLL-U file and returns a list of Sentence objects.
    Each sentence has a synthetic <root> token prepended at index 0.
    Multi-word tokens (IDs like '1-2') and empty nodes (IDs like '1.1') are skipped.
    """
    sentences = []

    with open(path, encoding='utf-8') as f:
        words   = ['<root>']
        lemmas  = ['<root>']
        upos    = ['<root>']
        xpos    = ['<root>']
        heads   = [0]
        deprels = ['root']

        for line in f:
            line = line.rstrip('\n')

            if line.startswith('#'):
                # comment line — skip
                continue

            if line == '':
                # blank line = sentence boundary
                if len(words) > 1:   # at least one real token
                    sentence = Sentence(
                        words=words, lemmas=lemmas, upos=upos, xpos=xpos,
                        heads=heads, deprels=deprels
                    )
                    # Build character lists (skip <root>)
                    sentence.chars = [[]] + [list(w) for w in words[1:]]
                    sentences.append(sentence)

                # Reset for next sentence
                words   = ['<root>']
                lemmas  = ['<root>']
                upos    = ['<root>']
                xpos    = ['<root>']
                heads   = [0]
                deprels = ['root']
                continue

            cols = line.split('\t')
            if len(cols) < 8:
                continue

            token_id = cols[0]
            # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
            if '-' in token_id or '.' in token_id:
                continue

            words.append(cols[1])
            lemmas.append(cols[2] if cols[2] != '_' else cols[1])
            upos.append(cols[3] if cols[3] != '_' else 'X')
            xpos.append(cols[4] if cols[4] != '_' else cols[3])

            head_val = cols[6]
            heads.append(int(head_val) if head_val.isdigit() else 0)

            deprels.append(cols[7] if cols[7] != '_' else 'dep')

        # Handle file without trailing newline
        if len(words) > 1:
            sentence = Sentence(
                words=words, lemmas=lemmas, upos=upos, xpos=xpos,
                heads=heads, deprels=deprels
            )
            sentence.chars = [[]] + [list(w) for w in words[1:]]
            sentences.append(sentence)

    return sentences


def write_conllu(sentences: List[Sentence], pred_heads: List[List[int]],
                 pred_rels: List[List[str]], path: str):
    """
    Writes predicted parse trees back to CoNLL-U format.
    pred_heads and pred_rels are parallel to sentences (index 0 = root, ignored).
    """
    with open(path, 'w', encoding='utf-8') as f:
        for sent, p_heads, p_rels in zip(sentences, pred_heads, pred_rels):
            for i in range(1, len(sent)):   # skip <root>
                f.write('\t'.join([
                    str(i),           # ID
                    sent.words[i],    # FORM
                    sent.lemmas[i],   # LEMMA
                    sent.upos[i],     # UPOS
                    sent.xpos[i],     # XPOS
                    '_',              # FEATS
                    str(p_heads[i]),  # HEAD (predicted)
                    p_rels[i],        # DEPREL (predicted)
                    '_',              # DEPS
                    '_',              # MISC
                ]) + '\n')
            f.write('\n')
