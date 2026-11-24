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
Corpus generation for the bank transfer schema.

Shared by ``02_seq_preproc_tokenization.ipynb`` and
``scripts/build_bank_corpus.py``:

    1. fit the tokenizer **once** on the training split
    2. persist ``tokenizer_state.json`` (+ ``vocab.json`` for training ranks)
    3. transform every split with that fitted tokenizer and write one text
       corpus per split

Step 1/2 matter: amount bins and the frequency vocabularies are data-driven,
so re-fitting per split would assign different IDs to the same value.
"""

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cudf

from .fingerprint import file_fingerprint
from .tokenizer.bank_pipeline import (
    BankSchema,
    BankTokenizerConfig,
    BankTransferTokenizerPipeline,
)

PathLike = Union[str, Path]


def read_table(path: PathLike, columns: Optional[List[str]] = None) -> cudf.DataFrame:
    """Read a parquet or CSV file into cuDF."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return cudf.read_parquet(str(path), columns=columns)
    if path.suffix.lower() in {".csv", ".gz", ".txt"}:
        return cudf.read_csv(str(path), usecols=columns)
    raise ValueError(f"Unsupported table format: {path.suffix} ({path})")


def fit_pipeline(
    train_path: PathLike,
    schema: Optional[BankSchema] = None,
    config: Optional[BankTokenizerConfig] = None,
    state_path: Optional[PathLike] = None,
    vocab_path: Optional[PathLike] = None,
    verbose: bool = True,
) -> Tuple[BankTransferTokenizerPipeline, cudf.DataFrame]:
    """Fit the tokenizer on the training split and optionally persist it.

    Returns the fitted pipeline and the preprocessed training frame (reused by
    the caller to avoid a second preprocess pass).
    """
    pipeline = BankTransferTokenizerPipeline(schema=schema, config=config)

    t0 = time.time()
    gdf = read_table(train_path)
    if verbose:
        print(f"  Loaded {len(gdf):,} training rows in {time.time() - t0:.1f}s")

    processed = pipeline.preprocess(gdf)
    pipeline.fit(processed)

    if verbose:
        print(f"  Fitted in {time.time() - t0:.1f}s")
        print(f"  {pipeline.describe()}")

    if state_path:
        print(f"  Tokenizer state -> {pipeline.save(state_path)}")
    if vocab_path:
        print(f"  Vocabulary      -> {pipeline.save_vocab(vocab_path)}")

    return pipeline, processed


def generate_corpus(
    pipeline: BankTransferTokenizerPipeline,
    out_path: PathLike,
    source: Union[PathLike, cudf.DataFrame],
    chunk_size: Optional[int] = None,
    context_window: int = 4096,
    verbose: bool = True,
    chunk_copies: int = 1,
) -> Dict[str, Any]:
    """Write one corpus line per (account, chunk) for a single split.

    *source* is a raw table path or an already-preprocessed cuDF frame.
    """
    if not pipeline.is_fitted:
        raise ValueError("Fit the pipeline on the training split first.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = chunk_size or pipeline.recommended_chunk_size(context_window)

    t0 = time.time()
    if isinstance(source, cudf.DataFrame):
        processed = source
    else:
        processed = pipeline.preprocess(read_table(source))

    token_df = pipeline.transform(processed)
    lines = pipeline.to_corpus_lines(
        token_df, processed, pipeline.group_cols, chunk_size=chunk_size,
        chunk_copies=chunk_copies,
    )

    with open(out_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    longest = max((len(line.split()) for line in lines), default=0)
    stats = {
        "rows": len(processed),
        "sequences": len(lines),
        "chunk_size": chunk_size,
        "chunk_copies": chunk_copies,
        "tokens_per_transaction": pipeline.tokens_per_transaction,
        "longest_sequence_tokens": longest,
        "path": str(out_path),
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(
            f"  {out_path.name}: {stats['sequences']:,} sequences from "
            f"{stats['rows']:,} transactions "
            f"(<= {longest} tokens, {stats['seconds']}s)"
        )
        if longest > context_window:
            print(
                f"  WARNING: longest sequence ({longest}) exceeds the "
                f"{context_window}-token context window and will be truncated "
                "during training. Lower chunk_size."
            )
    return stats


def build_all(
    splits: Iterable[Tuple[str, PathLike]],
    out_dir: PathLike,
    tokenizer_dir: PathLike,
    schema: Optional[BankSchema] = None,
    config: Optional[BankTokenizerConfig] = None,
    chunk_size: Optional[int] = None,
    context_window: int = 4096,
    chunk_copies: int = 1,
) -> Dict[str, Any]:
    """Fit on the split named ``train`` and write a corpus for every split.

    ``chunk_copies`` > 1 overlaps the *train* corpus only; the others keep one
    chunk per stretch of history, so validation loss stays comparable.
    """
    splits = list(splits)
    names = [name for name, _ in splits]
    if "train" not in names:
        raise ValueError(f"A split named 'train' is required, got {names}")

    out_dir, tokenizer_dir = Path(out_dir), Path(tokenizer_dir)
    train_path = dict(splits)["train"]

    print("Fitting tokenizer on the training split...")
    pipeline, train_processed = fit_pipeline(
        train_path,
        schema=schema,
        config=config,
        state_path=tokenizer_dir / "tokenizer_state.json",
        vocab_path=tokenizer_dir / "vocab.json",
    )

    print("\nGenerating corpora...")
    report: Dict[str, Any] = {
        "vocab_size": int(pipeline.global_vocab_size),
        "tokens_per_transaction": pipeline.tokens_per_transaction,
        "sequence_scope": pipeline.config.sequence_scope,
        "sequence_key": list(pipeline.group_cols),
        "chunk_size": chunk_size or pipeline.recommended_chunk_size(context_window),
        "context_window": context_window,
        "tokenizer_state": str(tokenizer_dir / "tokenizer_state.json"),
        # Content, not path: notebook 04 refuses a checkpoint trained on a
        # corpus whose tokenizer differs from the current one.
        "tokenizer_fingerprint": file_fingerprint(tokenizer_dir / "tokenizer_state.json"),
        "vocab_path": str(tokenizer_dir / "vocab.json"),
        "splits": {},
    }
    for name, path in splits:
        source = train_processed if name == "train" else path
        report["splits"][name] = generate_corpus(
            pipeline,
            out_dir / f"{name}_corpus.txt",
            source,
            chunk_size=chunk_size,
            context_window=context_window,
            chunk_copies=chunk_copies if name == "train" else 1,
        )

    return report
