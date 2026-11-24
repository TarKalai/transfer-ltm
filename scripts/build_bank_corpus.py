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

"""
Fit the bank-transfer tokenizer and write the pretraining corpora.

Notebook 02 does the same thing interactively; this script is the batch entry
point for larger extracts.

Example
-------
    python scripts/build_bank_corpus.py \
        --train data/bank/temporal_split/train.parquet \
        --val   data/bank/temporal_split/val.parquet \
        --test  data/bank/temporal_split/test.parquet \
        --out-dir data/bank/decoder_corpus \
        --tokenizer-dir data/bank/tokenizer

Then pretrain with the fitted vocabulary size:

    torchrun --nproc-per-node=8 scripts/train_bank_decoder.py \
        -c configs/pretrain_bank_decoder.yaml \
        --dataset.data_path data/bank/decoder_corpus/train_corpus.txt \
        --validation_dataset.data_path data/bank/decoder_corpus/val_corpus.txt \
        --model.config.vocab_size <vocab_size printed below>
"""

import argparse
import json
import sys
from pathlib import Path

BLUEPRINT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BLUEPRINT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="Training split (parquet or csv)")
    p.add_argument("--val", default=None, help="Validation split")
    p.add_argument("--test", default=None, help="Test split")
    p.add_argument("--out-dir", default="data/bank/decoder_corpus")
    p.add_argument("--tokenizer-dir", default="data/bank/tokenizer")
    p.add_argument("--context-window", type=int, default=4096)
    p.add_argument(
        "--chunk-size", type=int, default=None,
        help="Transactions per corpus line (default: fill the context window)",
    )
    p.add_argument(
        "--schema-json", default=None,
        help="JSON file with BankSchema overrides, e.g. "
             '{"amount": "amount_eur", "label": "is_fraud"}',
    )
    p.add_argument(
        "--config-json", default=None,
        help="JSON file with BankTokenizerConfig overrides, e.g. "
             '{"counterparty_hash_size": 8192, "amount_bins": 64}',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.bank_corpus import build_all
    from src.tokenizer.bank_pipeline import BankSchema, BankTokenizerConfig

    schema = BankSchema(**json.loads(Path(args.schema_json).read_text())) if args.schema_json else BankSchema()
    config = BankTokenizerConfig(**json.loads(Path(args.config_json).read_text())) if args.config_json else BankTokenizerConfig()

    splits = [("train", args.train)]
    if args.val:
        splits.append(("val", args.val))
    if args.test:
        splits.append(("test", args.test))

    report = build_all(
        splits=splits,
        out_dir=args.out_dir,
        tokenizer_dir=args.tokenizer_dir,
        schema=schema,
        config=config,
        chunk_size=args.chunk_size,
        context_window=args.context_window,
    )

    report_path = Path(args.out_dir) / "corpus_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 66)
    print(f"vocab_size              : {report['vocab_size']}")
    print(f"tokens per transaction  : {report['tokens_per_transaction']}")
    print(f"transactions per line   : {report['chunk_size']}")
    print(f"tokenizer state         : {report['tokenizer_state']}")
    print(f"vocabulary              : {report['vocab_path']}")
    print(f"report                  : {report_path}")
    print("=" * 66)
    print("\nPretrain with:")
    print(
        f"  torchrun --nproc-per-node=8 scripts/train_bank_decoder.py \\\n"
        f"      -c configs/pretrain_bank_decoder.yaml \\\n"
        f"      --dataset.data_path {report['splits']['train']['path']} \\\n"
        + (
            f"      --validation_dataset.data_path {report['splits']['val']['path']} \\\n"
            if "val" in report["splits"] else ""
        )
        + f"      --dataset.vocab_path {report['vocab_path']} \\\n"
        f"      --validation_dataset.vocab_path {report['vocab_path']} \\\n"
        f"      --model.config.vocab_size {report['vocab_size']}"
    )


if __name__ == "__main__":
    main()
