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

"""Bank Transfer Foundation Model Pretraining — Decoder-Only (plain PyTorch).

Command-line entry point for ``src/train_decoder.py`` (no ``nemo_automodel``):
the YAML plus ``--section.key value`` overrides. See ``src/train_decoder.py``
for why NeMo AutoModel was removed.

Launch
------
Single GPU::

    python scripts/train_bank_decoder.py \
        -c configs/pretrain_bank_decoder.yaml \
        --dataset.data_path data/bank/decoder_corpus/train_corpus.txt \
        --dataset.vocab_path data/bank/tokenizer/vocab.json \
        --validation_dataset.data_path data/bank/decoder_corpus/val_corpus.txt \
        --validation_dataset.vocab_path data/bank/tokenizer/vocab.json \
        --model.config.vocab_size <fitted vocab size from notebook 02> \
        --step_scheduler.max_steps 3000

Multi-GPU (DistributedDataParallel)::

    torchrun --nproc-per-node=8 scripts/train_bank_decoder.py \
        -c configs/pretrain_bank_decoder.yaml \
        ... same overrides ...
"""

import argparse
import json
import sys
from pathlib import Path

BLUEPRINT_ROOT = Path(__file__).resolve().parent.parent
if str(BLUEPRINT_ROOT) not in sys.path:
    sys.path.insert(0, str(BLUEPRINT_ROOT))


def main() -> int:
    from src.train_decoder import apply_overrides, load_config, train_bank_decoder

    parser = argparse.ArgumentParser(add_help=True, description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML config")
    parser.add_argument("--log-every-steps", type=int, default=10,
                        help="Print a training-loss line every N optimizer steps")
    parser.add_argument("--history-out", default=None,
                        help="Optional path to write the loss history as JSON")
    args, overrides = parser.parse_known_args()

    cfg = apply_overrides(load_config(args.config), overrides)
    history = train_bank_decoder(cfg, log_every_steps=args.log_every_steps)

    if args.history_out:
        Path(args.history_out).write_text(json.dumps(history, indent=2))
        print(f"Loss history -> {args.history_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())