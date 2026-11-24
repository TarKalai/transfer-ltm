# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Print exactly what the model is fed: one tokenized sequence and one transaction.

Reads a run's pretraining corpus (one line = one sequence) and its vocabulary,
so it needs a run whose stage 02 has finished::

    python scripts/show_tokens.py                 # run "base", first train sequence
    python scripts/show_tokens.py --run seed_1 --line 10 --transactions 5

Prints, for one sequence:

1. the sequence as the model reads it: ``<bos>``, transactions separated by
   ``<sep>``, ``<eos>`` (the first ``--transactions`` of them);
2. one transaction (``--pick``), token by token: field, token, and the integer
   ID that actually enters the model;
3. the ID list of the start of the sequence.

Notebook 04 feeds the model the same format at embedding time, with the last 32
transactions before the scored one instead of a pretraining chunk.

Reads only the run's files; nothing is written.
"""

import argparse
import json
import sys
from pathlib import Path

#: Token prefix -> what the field is, in plain words.
FIELDS = {
    "AMT": "amount (bin)",
    "AMTREL": "amount vs the customer's earlier amounts (bin)",
    "TDIF": "time since the customer's previous transfer (bin)",
    "TDIFA": "time since the previous transfer on this account (bin)",
    "DC": "debit / credit",
    "SIGN": "movement sign",
    "MVC": "movement code",
    "MVGRP": "movement code group",
    "CPTY": "counterparty account (own token or hash bucket)",
    "BIC": "counterparty BIC",
    "CTRY": "counterparty country",
    "OPST": "operations status",
    "CMT": "comment structure",
    "EMP": "employee (ordinal)",
    "TXT": "free text",
    "HOUR": "hour of day",
    "DOW": "day of week",
    "MONTH": "month",
    "ACCTH": "which of the customer's accounts (hash bucket)",
    "ACCT": "which of the customer's accounts (1 = first one)",
    "CUSTH": "customer (hash bucket)",
}


def _paths(run: str):
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from src.experiment import runs_root
    run_dir = runs_root() / run
    return run_dir / "decoder_corpus", run_dir / "tokenizer" / "vocab.json"


def _field(token: str) -> str:
    prefix = token.split("_", 1)[0]
    return FIELDS.get(prefix, prefix)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="base", help="run name under runs_dir (default: base)")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--line", type=int, default=0, help="which sequence (line) of the corpus")
    parser.add_argument("--transactions", type=int, default=3,
                        help="transactions of the sequence to print (default: 3)")
    parser.add_argument("--pick", type=int, default=0,
                        help="transaction shown token by token (default: the first)")
    args = parser.parse_args(argv)

    corpus_dir, vocab_path = _paths(args.run)
    corpus = corpus_dir / f"{args.split}_corpus.txt"
    for path in (corpus, vocab_path):
        if not path.exists():
            parser.error(f"{path} not found: run stage 02 of '{args.run}' first.")

    payload = json.loads(vocab_path.read_text())
    vocab = payload["vocab"]
    unk = vocab.get("<unk>", 4)

    with open(corpus) as f:
        for i, line in enumerate(f):
            if i == args.line:
                break
        else:
            parser.error(f"{corpus.name} has only {i + 1} lines")
    tokens = line.split()

    # <bos> t1 .. tT <sep> t1 .. tT <sep> ... <eos>
    body = tokens[1:-1] if tokens and tokens[-1] == "<eos>" else tokens[1:]
    txns, current = [], []
    for tok in body:
        if tok == "<sep>":
            txns.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        txns.append(current)

    print(f"Run {args.run!r}, {args.split} corpus, line {args.line}")
    print(f"  {len(tokens):,} tokens = <bos> + {len(txns)} transactions x "
          f"{len(txns[0]) if txns else 0} tokens + separators + <eos>")
    print(f"  vocabulary: {payload['vocab_size']:,} tokens\n")

    shown = txns[:args.transactions]
    print(f"1. THE SEQUENCE (first {len(shown)} of {len(txns)} transactions)\n")
    print("  <bos>")
    for n, txn in enumerate(shown):
        print(f"  {' '.join(txn)}")
        print("  <sep>" if n < len(txns) - 1 else "  <eos>")
    if len(txns) > len(shown):
        print(f"  ... {len(txns) - len(shown)} more transactions ... <eos>")

    if not txns:
        return 0
    pick = min(args.pick, len(txns) - 1)
    print(f"\n2. ONE TRANSACTION (number {pick}), token by token\n")
    print(f"  {'position':>8}  {'token':<22} {'ID':>6}  field")
    for pos, tok in enumerate(txns[pick], 1):
        tid = vocab.get(tok, unk)
        print(f"  {pos:>8}  {tok:<22} {tid:>6}  {_field(tok)}")
    sep = vocab.get("<sep>", 3)
    print(f"  {'':>8}  {'<sep>':<22} {sep:>6}  end of transaction")

    head = tokens[: 1 + 2 * (len(txns[0]) + 1)]
    ids = [vocab.get(t, unk) for t in head]
    print(f"\n3. WHAT ENTERS THE MODEL: integer IDs (first {len(ids)} of {len(tokens):,})\n")
    print(f"  {ids}")
    print("\n  The model only ever sees these integers; the names above are for reading.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
