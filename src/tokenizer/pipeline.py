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
Modular tokenizer pipeline for causal language modeling.

Orchestrates multiple BaseTokenizer steps, manages a global vocabulary with
per-step offsets, and provides corpus generation / encoding utilities.

Special tokens: <bos> / <eos> / <sep> / <pad> / <unk>
Default chunk_size: ~315 transactions for a 4096-token context window.
Sequence format: "<bos> txn1 <sep> txn2 <sep> ... txnN <eos>"
"""

import json
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import cudf
import cupy as cp
import numpy as np

from .base import BaseTokenizer
from .registry import resolve_tokenizer

STATE_FORMAT = "tokenizer_pipeline_v1"


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy/cupy values in a state dict to plain Python."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, cp.ndarray):
        return obj.get().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _int_keys(mapping: Any) -> Any:
    """Restore integer keys lost to JSON's string-only object keys."""
    if not isinstance(mapping, dict):
        return mapping
    if mapping and all(isinstance(k, str) and k.lstrip("-").isdigit() for k in mapping):
        return {int(k): _int_keys(v) for k, v in mapping.items()}
    return {k: _int_keys(v) for k, v in mapping.items()}


class TokenizerPipeline:
    """Manages an ordered sequence of BaseTokenizer steps with global vocab offsets."""

    SPECIAL_TOKENS = {
        "PAD": "<pad>",
        "BOS": "<bos>",
        "EOS": "<eos>",
        "SEP": "<sep>",
        "UNK": "<unk>",
    }

    #: Identifier of the string-hash function this pipeline's ``preprocess()``
    #: uses to build hash-bucket columns, or ``None`` if it hashes nothing.
    #: Recorded in the state file and checked on load — a hash-bucket token
    #: means "the value landing in this bucket under *this* hash", so reusing a
    #: corpus or checkpoint under a different one is silently wrong.
    HASH_ALGO: ClassVar[Optional[str]] = None
    #: What a state without a ``hash_algo`` key was hashed with, when that is
    #: certain (TabFormer always used cuDF's hash_values). None: unknown, so
    #: such states are refused.
    LEGACY_HASH_ALGO: ClassVar[Optional[str]] = None

    def __init__(
        self,
        special_tokens: Optional[Dict[str, str]] = None,
        use_streams: bool = True,
        stream_threshold: int = 5,
    ):
        self.steps: Dict[str, BaseTokenizer] = {}
        self.vocab_sizes: Dict[str, int] = {}
        self.vocab_offset: Dict[str, int] = {}
        self.column_specs: Dict[str, List[str]] = {}
        self.tokenizer_order: List[str] = []
        self.stream_threshold = stream_threshold

        self.special_tokens = special_tokens or dict(self.SPECIAL_TOKENS)
        self.special_token_ids = {
            tok: idx for idx, tok in enumerate(self.special_tokens.values())
        }
        self.num_special_tokens = len(self.special_tokens)
        self.global_vocab_size = self.num_special_tokens

        self.use_streams = use_streams
        self.is_fitted = False

        self._vocab: Optional[Dict[str, int]] = None
        self._id_to_token: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_key(col):
        return list(col) if isinstance(col, list) else [col]

    def add_step(
        self, column_name: Union[list, str], tokenizer: BaseTokenizer
    ) -> "TokenizerPipeline":
        col = self._normalize_key(column_name)
        tok_id = "_".join(col)
        self.steps[tok_id] = tokenizer
        self.column_specs[tok_id] = col
        self.tokenizer_order.append(tok_id)
        return self

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: cudf.DataFrame) -> "TokenizerPipeline":
        if self.use_streams and len(self.steps) >= self.stream_threshold:
            self._fit_parallel(df)
        else:
            self._fit_sequential(df)
        self._build_global_vocab()
        self.is_fitted = True
        return self

    def _fit_sequential(self, df: cudf.DataFrame) -> None:
        current_offset = self.num_special_tokens
        for tok_id in self.tokenizer_order:
            columns = self.column_specs[tok_id]
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame")

            tokenizer = self.steps[tok_id]
            col_data = df[columns] if len(columns) > 1 else df[columns[0]]
            tokenizer.build_vocab(col_data)

            vs = tokenizer.vocab_size
            self.vocab_sizes[tok_id] = vs

            if tokenizer._idx_to_token and isinstance(
                next(iter(tokenizer._idx_to_token.values())), dict
            ):
                for sub in tokenizer._idx_to_token:
                    sub_size = len(tokenizer._idx_to_token[sub])
                    self.vocab_offset[f"{tok_id}.{sub}"] = current_offset
                    current_offset += sub_size
            else:
                self.vocab_offset[tok_id] = current_offset
                current_offset += vs

        self.global_vocab_size = current_offset

    def _fit_parallel(self, df: cudf.DataFrame) -> None:
        streams = [
            cp.cuda.Stream(non_blocking=True) for _ in self.tokenizer_order
        ]
        for stream, tok_id in zip(streams, self.tokenizer_order):
            columns = self.column_specs[tok_id]
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame")

            tokenizer = self.steps[tok_id]
            col_data = df[columns] if len(columns) > 1 else df[columns[0]]
            original_stream = getattr(tokenizer, "stream", None)
            if hasattr(tokenizer, "stream"):
                tokenizer.stream = stream
            with stream:
                tokenizer.build_vocab(col_data)
            if hasattr(tokenizer, "stream"):
                tokenizer.stream = original_stream

        cp.cuda.Stream.null.synchronize()

        current_offset = self.num_special_tokens
        for tok_id in self.tokenizer_order:
            tokenizer = self.steps[tok_id]
            vs = tokenizer.vocab_size
            self.vocab_sizes[tok_id] = vs

            if tokenizer._idx_to_token and isinstance(
                next(iter(tokenizer._idx_to_token.values())), dict
            ):
                for sub in tokenizer._idx_to_token:
                    sub_size = len(tokenizer._idx_to_token[sub])
                    self.vocab_offset[f"{tok_id}.{sub}"] = current_offset
                    current_offset += sub_size
            else:
                self.vocab_offset[tok_id] = current_offset
                current_offset += vs

        self.global_vocab_size = current_offset

    # ------------------------------------------------------------------
    # Global vocab helpers
    # ------------------------------------------------------------------

    def _build_global_vocab(self) -> None:
        vocab: Dict[str, int] = {}
        id_to_token: Dict[int, str] = {}

        for tok_str, idx in self.special_token_ids.items():
            vocab[tok_str] = idx
            id_to_token[idx] = tok_str

        for tok_id in self.tokenizer_order:
            tokenizer = self.steps[tok_id]
            if tokenizer._idx_to_token is None:
                continue

            if isinstance(next(iter(tokenizer._idx_to_token.values()), None), dict):
                for sub_key, sub_dict in tokenizer._idx_to_token.items():
                    offset = self.vocab_offset.get(f"{tok_id}.{sub_key}", 0)
                    for local_idx, token_str in sub_dict.items():
                        gid = int(local_idx) + offset
                        vocab[token_str] = gid
                        id_to_token[gid] = token_str
            else:
                offset = self.vocab_offset.get(tok_id, 0)
                for local_idx, token_str in tokenizer._idx_to_token.items():
                    gid = int(local_idx) + offset
                    vocab[token_str] = gid
                    id_to_token[gid] = token_str

        self._vocab = vocab
        self._id_to_token = id_to_token

        collisions = self.id_collisions()
        if collisions:
            example = ", ".join(
                f"{a}/{b} -> {gid}" for gid, (a, b) in list(collisions.items())[:3]
            )
            warnings.warn(
                f"{len(collisions)} token ID collision(s) in the global vocabulary "
                f"({example}). A step's local indices overrun its vocabulary slot -- "
                "the usual cause is FixedVocabTokenizer with min_val > 0 and "
                "zero_based_index=False, which keys the vocabulary by value. The "
                "colliding tokens share one embedding row and one of them cannot be "
                "decoded.",
                stacklevel=3,
            )

    def id_collisions(self) -> Dict[int, tuple]:
        """``{global_id: (token_a, token_b)}`` for IDs claimed by two tokens."""
        if not self._vocab:
            return {}
        seen: Dict[int, str] = {}
        clashes: Dict[int, tuple] = {}
        for token, gid in self._vocab.items():
            if gid in seen and seen[gid] != token:
                clashes[gid] = (seen[gid], token)
            else:
                seen[gid] = token
        return clashes

    def field_prefixes(self) -> List[str]:
        """Token prefix of every step (``AMT``, ``CPTY``, ``TXT``, ...)."""
        return [self.steps[t].token_prefix() for t in self.tokenizer_order]

    @property
    def field_names(self) -> List[str]:
        """Token positions of one transaction, in emission order.

        Equal to ``tokenizer_order`` unless a step emits several tokens, in
        which case its output columns are listed in its place.
        """
        names: List[str] = []
        for tok_id in self.tokenizer_order:
            names.extend(self.steps[tok_id].output_columns(tok_id))
        return names

    @property
    def field_vocab_sizes(self) -> Dict[str, int]:
        """``{field name: vocabulary size of the step that emits it}``."""
        sizes: Dict[str, int] = {}
        for tok_id in self.tokenizer_order:
            for name in self.steps[tok_id].output_columns(tok_id):
                sizes[name] = int(self.vocab_sizes.get(tok_id, 0))
        return sizes

    @property
    def vocab(self) -> Dict[str, int]:
        if self._vocab is None:
            return {}
        return self._vocab

    @property
    def id_to_token(self) -> Dict[int, str]:
        if self._id_to_token is None:
            return {}
        return self._id_to_token

    # ------------------------------------------------------------------
    # Transform  ->  DataFrame of token strings
    # ------------------------------------------------------------------

    def transform(self, df: cudf.DataFrame) -> cudf.DataFrame:
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")

        local_results = {}
        if self.use_streams and len(self.steps) >= self.stream_threshold:
            streams = [
                cp.cuda.Stream(non_blocking=True) for _ in self.tokenizer_order
            ]
            for stream, tok_id in zip(streams, self.tokenizer_order):
                columns = self.column_specs[tok_id]
                col_data = df[columns] if len(columns) > 1 else df[columns[0]]
                tokenizer = self.steps[tok_id]
                if hasattr(tokenizer, "stream"):
                    original = tokenizer.stream
                    tokenizer.stream = stream
                with stream:
                    local_results[tok_id] = tokenizer.tokenize(col_data)
                if hasattr(tokenizer, "stream"):
                    tokenizer.stream = original
            cp.cuda.Stream.null.synchronize()
        else:
            for tok_id in self.tokenizer_order:
                columns = self.column_specs[tok_id]
                col_data = df[columns] if len(columns) > 1 else df[columns[0]]
                local_results[tok_id] = self.steps[tok_id].tokenize(col_data)

        parts = []
        for key, val in local_results.items():
            if isinstance(val, cudf.Series):
                parts.append(val.to_frame(name=key))
            elif isinstance(val, cudf.DataFrame):
                parts.append(val)
            else:
                parts.append(cudf.DataFrame({key: val}))

        return cudf.concat(parts, axis=1)

    def fit_transform(self, df: cudf.DataFrame) -> cudf.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Encode  ->  padded integer-ID numpy array (causal LM format)
    # ------------------------------------------------------------------

    def encode(
        self,
        token_df: cudf.DataFrame,
        max_length: int = 4096,
        add_special: bool = True,
    ) -> np.ndarray:
        """Convert a token-string DataFrame to a padded int64 array.

        Each row becomes: <bos> col1 col2 ... colN <eos> <pad> ...
        Returns shape (n_rows, max_length).
        """
        vocab = self._vocab
        pad_id = self.special_token_ids["<pad>"]
        bos_id = self.special_token_ids["<bos>"]
        eos_id = self.special_token_ids["<eos>"]
        unk_id = self.special_token_ids["<unk>"]

        n = len(token_df)
        padded = np.full((n, max_length), pad_id, dtype=np.int64)

        col_offset = 0
        if add_special:
            padded[:, 0] = bos_id
            col_offset = 1

        for col_name in token_df.columns:
            if col_offset >= max_length:
                break
            host_col = token_df[col_name].to_pandas()
            ids = host_col.map(vocab).fillna(unk_id).astype(np.int64).values
            padded[:, col_offset] = ids
            col_offset += 1

        if add_special and col_offset < max_length:
            padded[:, col_offset] = eos_id

        return padded

    # ------------------------------------------------------------------
    # Corpus generation  ->  list of "<bos> t1 <sep> t2 ..." lines
    # ------------------------------------------------------------------

    def to_corpus_lines(
        self,
        token_df: cudf.DataFrame,
        df_meta: cudf.DataFrame,
        group_cols: List[str],
        chunk_size: int = 315,
        chunk_copies: int = 1,
    ) -> List[str]:
        """Assemble token strings into text corpus lines for causal LM training.

        Parameters
        ----------
        token_df : cudf.DataFrame
            Output of transform() -- one column per token field.
        df_meta : cudf.DataFrame
            Original (preprocessed) DataFrame with *group_cols* for grouping.
        group_cols : list of str
            Columns to group transactions by (e.g. ["user", "card"]).
        chunk_size : int
            Max transactions per sequence (315 fits ~4096 tokens with
            12 tokens/txn + separators).
        chunk_copies : int
            1 (default): consecutive, non-overlapping chunks. k > 1: chunks of
            ``k * (chunk_size // k)`` transactions (never more than
            ``chunk_size``, so they still fit the context) start every
            ``chunk_size // k`` transactions: except in the first and last k-1
            blocks of a sequence, each transaction appears in exactly k chunks,
            with different amounts of history before it. A chunk entirely
            contained in the previous one is dropped.
        """
        token_cols = list(token_df.columns)
        txn_text = token_df[token_cols[0]].str.cat(
            [token_df[c] for c in token_cols[1:]], sep=" "
        )

        work = df_meta[group_cols].copy()
        work["_txn_text"] = txn_text

        work["_seq_id"] = work.groupby(group_cols).cumcount()
        if chunk_copies > 1:
            return self._overlapping_lines(work, group_cols, chunk_size, chunk_copies)
        work["_chunk_id"] = (work["_seq_id"] // chunk_size).astype("int32")

        grouped = work.groupby(group_cols + ["_chunk_id"])["_txn_text"].agg(list)
        if hasattr(grouped, "to_pandas"):
            grouped = grouped.to_pandas()

        def _fmt(txn_list):
            return "<bos> " + " <sep> ".join(txn_list) + " <eos>"

        return grouped.map(_fmt).tolist()

    @staticmethod
    def _overlapping_lines(work, group_cols, chunk_size: int, copies: int) -> List[str]:
        """Corpus lines with overlapping chunks (see ``chunk_copies``).

        Transactions are first joined into blocks of ``stride`` (one row per
        block, a few hundred times fewer rows), then chunk k of a sequence is
        blocks k .. k + copies - 1. Memory stays near one copy of the text,
        instead of ``copies`` copies of every row.
        """
        if copies > chunk_size:
            raise ValueError(f"chunk_copies={copies} exceeds the chunk size ({chunk_size} "
                             "transactions): chunks would overrun the context window")
        stride = chunk_size // copies
        work["_block"] = (work["_seq_id"] // stride).astype("int32")
        blocks = work.groupby(group_cols + ["_block"])["_txn_text"].agg(list)
        blocks = blocks.to_pandas() if hasattr(blocks, "to_pandas") else blocks
        blocks = blocks.map(" <sep> ".join).rename("text").reset_index()
        blocks = blocks.sort_values(group_cols + ["_block"]).reset_index(drop=True)

        by_seq = blocks.groupby(group_cols, sort=False)["text"]
        parts = [blocks["text"]] + [by_seq.shift(-j) for j in range(1, copies)]
        # Chunk k (k > 0) is kept only if it reaches a block chunk k-1 lacks.
        keep = (blocks["_block"] == 0) | parts[-1].notna()
        lines = []
        for row in zip(*(p[keep] for p in parts)):
            body = " <sep> ".join(x for x in row if isinstance(x, str))
            lines.append("<bos> " + body + " <eos>")
        return lines

    # ------------------------------------------------------------------
    # Serialization  —  fit once on train, reuse for val / test / inference
    # ------------------------------------------------------------------

    def state_metadata(self) -> Dict[str, Any]:
        """Hook for subclasses to persist their own configuration."""
        return {}

    def get_state(self) -> Dict[str, Any]:
        """Full pipeline state: step configs, fitted vocabularies and offsets."""
        return {
            "format": STATE_FORMAT,
            "pipeline_class": type(self).__name__,
            "hash_algo": type(self).HASH_ALGO,
            "special_tokens": dict(self.special_tokens),
            "tokenizer_order": list(self.tokenizer_order),
            # Per token position, for diagnostics that only read this JSON.
            "field_names": self.field_names,
            "field_vocab": self.field_vocab_sizes,
            "column_specs": {k: list(v) for k, v in self.column_specs.items()},
            "vocab_sizes": dict(self.vocab_sizes),
            "vocab_offset": dict(self.vocab_offset),
            "global_vocab_size": int(self.global_vocab_size),
            "is_fitted": bool(self.is_fitted),
            "steps": {
                tok_id: _jsonify(self.steps[tok_id].get_state())
                for tok_id in self.tokenizer_order
            },
            "metadata": _jsonify(self.state_metadata()),
        }

    def _assert_no_collisions(self, what: str) -> None:
        collisions = self.id_collisions()
        if collisions:
            raise ValueError(
                f"Refusing to write {what}: {len(collisions)} token ID collision(s), "
                f"e.g. {list(collisions.items())[:3]}. Two tokens sharing an ID means "
                "one of them can never be decoded and both share an embedding row. "
                "Give the offending FixedVocabTokenizer step zero_based_index=True."
            )

    def save(self, path: Union[str, Path]) -> Path:
        """Write the fitted pipeline to a JSON state file."""
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted; call fit() before save().")
        self._assert_no_collisions("the tokenizer state")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.get_state()
        try:
            payload = json.dumps(state, indent=2)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError(
                "Pipeline state is not JSON-serializable. A step is holding an "
                "object (typically a fitted estimator) in its fitted state; "
                "store plain numbers instead, the way QuantileBinTokenizer "
                f"stores bin edges. Original error: {exc}"
            ) from exc
        path.write_text(payload)
        return path

    def save_vocab(self, path: Union[str, Path]) -> Path:
        """Write a standalone ``{token: id}`` vocabulary file.

        Used by the training data module (``src/clm_data.py``), which only
        needs to map corpus token strings to IDs and therefore should not
        have to import cuDF or rebuild the tokenizer steps.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted; call fit() before save_vocab().")
        self._assert_no_collisions("the vocabulary file")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "vocab_size": int(
                        max(self.global_vocab_size, max(self.vocab.values(), default=0) + 1)
                    ),
                    "special_tokens": dict(self.special_tokens),
                    "vocab": dict(self.vocab),
                    # For per-field loss weights: the token prefix of every
                    # field, and the tokens that only pad.
                    "field_prefixes": self.field_prefixes(),
                    "filler_tokens": sorted(
                        t for s in self.steps.values() for t in s.filler_tokens()),
                },
                indent=2,
            )
        )
        return path

    def _assert_hash_algo(self, state: Dict[str, Any]) -> None:
        """Refuse a state file written under a different string-hash function.

        Checked in one direction only, which is where the danger is: a pipeline
        that *hashes* (its ``preprocess()`` builds hash-bucket columns) must not
        tokenize new data against a vocabulary fitted under another hash, or
        ``CPTY_1183`` silently comes to mean a different counterparty. A
        vocabulary-only view — ``TokenizerPipeline.from_state`` with
        ``HASH_ALGO = None`` — hashes nothing and is safe to build from any
        state.
        """
        expected = type(self).HASH_ALGO
        if expected is None:
            return
        found = state.get("hash_algo")
        if found is None and state.get("pipeline_class") == type(self).__name__:
            # An unkeyed state that names this very pipeline class (never
            # another's, and never one that records no class at all).
            found = type(self).LEGACY_HASH_ALGO
        if found == expected:
            return
        detail = (
            "predates the hash_algo guard (written before the hash was pinned)"
            if found is None
            else f"was written with hash_algo={found!r}"
        )
        raise ValueError(
            f"Tokenizer state {detail}, but {type(self).__name__} hashes with "
            f"{expected!r}. Hash-bucket tokens (CPTY_*, CUSTH_*, ACCTH_*) are "
            "defined as hash % vocab_limit, so a different hash silently "
            "reassigns every bucket: the corpus, this vocabulary and any "
            "checkpoint trained on them would disagree about what each token "
            "means, without raising an error.\n"
            "Re-fit the tokenizer and regenerate the corpora (notebook 02 or "
            "scripts/build_bank_corpus.py), then retrain."
        )

    def _load_steps(self, state: Dict[str, Any], strict: bool = True) -> None:
        """Restore fitted step state, vocab offsets and the global vocabulary."""
        if state.get("format") != STATE_FORMAT:
            raise ValueError(
                f"Unsupported tokenizer state format: {state.get('format')!r}"
            )

        self._assert_hash_algo(state)

        order = list(state["tokenizer_order"])
        if strict and self.tokenizer_order and self.tokenizer_order != order:
            raise ValueError(
                "Saved pipeline steps do not match this pipeline's configuration.\n"
                f"  saved   : {order}\n"
                f"  current : {self.tokenizer_order}\n"
                "Rebuild the pipeline with the same schema/config used at fit time."
            )

        self.steps = {}
        self.tokenizer_order = order
        self.column_specs = {k: list(v) for k, v in state["column_specs"].items()}

        for tok_id in order:
            step_state = dict(state["steps"][tok_id])
            vocab_state = dict(step_state.get("vocab_state") or {})
            vocab_state["_idx_to_token"] = _int_keys(vocab_state.get("_idx_to_token"))
            step_state["vocab_state"] = vocab_state
            cls = resolve_tokenizer(step_state["class"])
            self.steps[tok_id] = cls.from_state(step_state)

        self.special_tokens = dict(state["special_tokens"])
        self.special_token_ids = {
            tok: idx for idx, tok in enumerate(self.special_tokens.values())
        }
        self.num_special_tokens = len(self.special_tokens)
        self.vocab_sizes = {k: int(v) for k, v in state["vocab_sizes"].items()}
        self.vocab_offset = {k: int(v) for k, v in state["vocab_offset"].items()}
        self.global_vocab_size = int(state["global_vocab_size"])
        self.is_fitted = bool(state.get("is_fitted", True))
        self._build_global_vocab()

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TokenizerPipeline":
        """Rebuild a bare pipeline (steps + vocabulary) from a state dict.

        Subclasses that add ``preprocess()`` logic should override this to
        restore their own configuration as well.
        """
        pipeline = TokenizerPipeline(special_tokens=dict(state["special_tokens"]))
        pipeline._load_steps(state, strict=False)
        return pipeline

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TokenizerPipeline":
        """Load a pipeline from a JSON state file written by :meth:`save`."""
        state = json.loads(Path(path).read_text())
        return cls.from_state(state)

