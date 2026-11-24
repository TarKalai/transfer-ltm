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
Tokenizer interface for decoder training and inference on bank transfer data.

Public API: ``encode``, ``decode``, ``vocab_size``, ``*_token_id``, as used by
``FinancialCLMDataset`` and ``HuggingFaceDecoderInference``.

The difference: this tokenizer has **no static vocabulary**.  Amount bins and
the frequency vocabularies are fitted on the training split, so it must be
built from the state file written by
``BankTransferTokenizerPipeline.save()``::

    tokenizer = BankTransactionTokenizer.from_state_file(
        "data/bank/tokenizer_state.json"
    )

Building it any other way would produce token IDs that disagree with the
training corpus.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .bank_pipeline import BankTransferTokenizerPipeline


class BankTransactionTokenizer:
    """Vocabulary view over a fitted :class:`BankTransferTokenizerPipeline`."""

    def __init__(self, pipeline: BankTransferTokenizerPipeline):
        if not pipeline.is_fitted:
            raise ValueError(
                "BankTransactionTokenizer requires a fitted pipeline. Fit on the "
                "training split and save the state, or use from_state_file()."
            )
        self._pipeline = pipeline
        self.vocab: Dict[str, int] = dict(pipeline.vocab)
        self.id_to_token: Dict[int, str] = dict(pipeline.id_to_token)

        self.special_tokens = {
            "pad": "<pad>",
            "bos": "<bos>",
            "eos": "<eos>",
            "sep": "<sep>",
            "unk": "<unk>",
        }
        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.bos_token_id = self.vocab.get("<bos>", 1)
        self.eos_token_id = self.vocab.get("<eos>", 2)
        self.sep_token_id = self.vocab.get("<sep>", 3)
        self.unk_token_id = self.vocab.get("<unk>", 4)
        self.special_token_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.sep_token_id,
            self.unk_token_id,
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_state_file(
        cls, path: Union[str, Path]
    ) -> "BankTransactionTokenizer":
        """Load from the JSON state written by ``pipeline.save(path)``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer state not found: {path}. Run notebook 02 "
                "(or scripts/build_bank_corpus.py) to fit and save it."
            )
        state = json.loads(path.read_text())
        return cls(BankTransferTokenizerPipeline.from_state(state))

    @property
    def pipeline(self) -> BankTransferTokenizerPipeline:
        return self._pipeline

    @property
    def tokens_per_transaction(self) -> int:
        return self._pipeline.tokens_per_transaction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), self._pipeline.global_vocab_size)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        tokens = self.tokenize(text)
        if max_length:
            tokens = tokens[:max_length]
            while len(tokens) < max_length:
                tokens.append("<pad>")
        unk = self.unk_token_id
        return [self.vocab.get(t, unk) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for tid in token_ids:
            tok = self.id_to_token.get(int(tid))
            if tok and tok != "<pad>":
                tokens.append(tok)
        return " ".join(tokens)
