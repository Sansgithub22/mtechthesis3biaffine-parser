# compare_results.py
# Print a full comparison table of all models:
#   - Our custom BiLSTM baseline
#   - Our BiLSTM + CharLSTM (Innovation 1)
#   - Our BERT parser (Innovation 2)
#   - Trankit pre-trained (XLM-RoBERTa, zero-shot on EWT)
#   - Trankit fine-tuned  (XLM-RoBERTa, trained on EWT)
#
# Usage:
#   python compare_results.py
#
# To update any number, just edit the RESULTS dict below after running
# eval_pretrained.py and eval_finetuned.py.

RESULTS = {
    # ---- Our custom implementations (biaffine_parser/) ----
    'Baseline BiLSTM\n(Dozat & Manning 2017)': {
        'params':  '12.4M',
        'epochs':  30,
        'dev_las': 87.43,
        'uas':     89.44,
        'las':     87.41,
    },
    'BiLSTM + CharLSTM\n[Our Innovation 1]': {
        'params':  '12.8M',
        'epochs':  25,
        'dev_las': 88.36,
        'uas':     89.68,
        'las':     87.92,
    },
    'BERT (bert-base-uncased)\n[Our Innovation 2]': {
        'params':  '111M',
        'epochs':  17,
        'dev_las': 92.71,
        'uas':     94.49,
        'las':     92.62,
    },
    # ---- Trankit models ----
    'Trankit Pre-trained\n(XLM-RoBERTa, zero-shot)': {
        'params':  '278M',
        'epochs':  'pre-trained',
        'dev_las': None,
        'uas':     None,   # fill in after running eval_pretrained.py
        'las':     None,
    },
    'Trankit Fine-tuned\n(XLM-RoBERTa + EWT)': {
        'params':  '278M',
        'epochs':  10,
        'dev_las': None,
        'uas':     None,   # fill in after running eval_finetuned.py
        'las':     None,
    },
}


def fmt(val, suffix='%'):
    if val is None:
        return 'pending'
    if isinstance(val, float):
        return f'{val:.2f}{suffix}'
    return str(val)


def print_table():
    header = f"{'Model':<42} {'Params':>8} {'Epochs':>10} {'Dev LAS':>9} {'Test UAS':>10} {'Test LAS':>10}"
    sep    = '-' * len(header)

    print(f'\n{"="*len(header)}')
    print('  FULL RESULTS COMPARISON — UD English EWT')
    print(f'{"="*len(header)}')
    print(header)
    print(sep)

    baseline_las = RESULTS[list(RESULTS.keys())[0]]['las']

    for name, r in RESULTS.items():
        display_name = name.replace('\n', ' ')
        las = r['las']
        gain = ''
        if las is not None and baseline_las is not None and las != baseline_las:
            diff = las - baseline_las
            gain = f'  (+{diff:.2f}%)' if diff > 0 else f'  ({diff:.2f}%)'

        print(f"{display_name:<42} {r['params']:>8} {fmt(r['epochs'], ''):>10} "
              f"{fmt(r['dev_las']):>9} {fmt(r['uas']):>10} {fmt(r['las'])}{gain}")

    print(sep)
    print()
    print('Notes:')
    print('  - Trankit uses XLM-RoBERTa-large (278M params) as encoder')
    print('  - "pending" = run eval_pretrained.py / eval_finetuned.py to fill in')
    print('  - Innovations 3+ will be added on top of the Trankit fine-tuned model')
    print()


if __name__ == '__main__':
    print_table()
