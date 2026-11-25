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
One experiment = one YAML file = one run folder.

``scripts/run_experiment.py experiments/<name>.yaml`` executes notebooks 02-05
headless with ``LTM_EXPERIMENT`` pointing at the run's frozen copy of the
YAML. Each notebook calls :func:`active_experiment`; when it returns an
experiment, the notebook takes its paths and settings from it. Run
interactively (variable unset), the notebooks behave exactly as before.

YAML sections (all optional except ``name``; see docs/experiments.md)::

    name: rare_bins_mean          # run folder: <runs root>/<name>
    description: ...
    base: base.yaml               # inherit, then override (deep merge)
    from_run: other_run           # reuse that run's tokenizer, corpus and model
    schema:     {description: "comm_text"}           # BankSchema fields
    tokenizer:  {tail_strategy: hash, ...}           # BankTokenizerConfig fields
    corpus:     {context_window: 2048}               # NB02: pretraining sequence length
    training:   {step_scheduler.max_steps: 2000}     # pretrain YAML keys, for NB03
    embeddings: {pooling: mean, history_window: 32}  # NB04 settings
    xgboost:    {pca_dim: 64, params: {max_depth: 6}}

Locations come from ``configs/paths.yaml`` -- the one place to set them
(raw extract, prepared data, run folders). The environment variables
``LTM_RAW_FILE``, ``LTM_DATA_DIR`` and ``LTM_RUNS_ROOT`` override it if set
(used by the offline tests; not needed otherwise).
"""

import copy
import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_EXPERIMENT = "LTM_EXPERIMENT"
ENV_DATA_DIR = "LTM_DATA_DIR"
ENV_RUNS_ROOT = "LTM_RUNS_ROOT"

SECTIONS = {"name", "description", "base", "from_run", "schema", "tokenizer",
            "corpus", "training", "embeddings", "xgboost"}

#: Experiment key -> notebook 02 variable.
CORPUS_SETTINGS = {"context_window": "CONTEXT_WINDOW", "chunk_copies": "CHUNK_COPIES",
                   "pretrain_on_test": "PRETRAIN_ON_TEST"}

#: Experiment key -> notebook 04 variable.
EMBEDDING_SETTINGS = {
    "pooling": "POOLING", "history_window": "HISTORY_WINDOW", "batch_size": "BATCH_SIZE",
    "history_context": "HISTORY_CONTEXT", "restrict_to_sequences": "RESTRICT_TO_SEQUENCES",
    "head_fraction": "HEAD_FRACTION", "head_max_negatives": "HEAD_MAX_NEGATIVES",
    "head_seed": "HEAD_SEED", "finetune": "FINETUNE", "finetune_lr": "FINETUNE_LR",
    "finetune_epochs": "FINETUNE_EPOCHS", "finetune_batch": "FINETUNE_BATCH",
    "labels": "LABELS",
}
TRAINING_YAML = PROJECT_ROOT / "configs" / "pretrain_bank_decoder.yaml"
#: Training keys the trainer reads that the YAML leaves commented out.
OPTIONAL_TRAINING_KEYS = {"step_scheduler.val_batches"}

#: Experiment key -> notebook 05 variable ("params" is merged into XGB_PARAMS).
XGBOOST_SETTINGS = {"use_pca": "USE_PCA", "pca_dim": "PCA_DIM", "random_state": "RANDOM_STATE",
                    "logreg": "LOGREG"}


PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yaml"
ENV_RAW_FILE = "LTM_RAW_FILE"


def _configured(key: str, env: str, default: str) -> Path:
    """A location: environment override, else configs/paths.yaml, else default.

    Relative paths are relative to the code folder, wherever Python runs from.
    """
    value = os.environ.get(env)
    if not value and PATHS_FILE.exists():
        value = (yaml.safe_load(PATHS_FILE.read_text()) or {}).get(key)
    path = Path(value or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


def raw_file() -> Path:
    """The raw extract (configs/paths.yaml: raw_file)."""
    return _configured("raw_file", ENV_RAW_FILE, "data/bank/raw/transfers.parquet")


def shared_data_dir(default: Optional[Path] = None) -> Path:
    """Prepared data: splits and schema (configs/paths.yaml: data_dir)."""
    return _configured("data_dir", ENV_DATA_DIR, str(default or "data/bank"))


def runs_root() -> Path:
    """One folder per experiment (configs/paths.yaml: runs_dir)."""
    return _configured("runs_dir", ENV_RUNS_ROOT, "data/runs")


#: Mappings a child replaces whole instead of merging into: a child's
#: ``loss_weights: {CPTY: 2}`` means exactly that, not the base's weights plus CPTY.
REPLACED_WHOLE = {"loss_weights"}


def _merge(base: Dict, over: Dict) -> Dict:
    """Child over base, recursively (``xgboost.params.eta`` keeps the base's
    ``max_depth``), except for the keys in ``REPLACED_WHOLE``."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if (isinstance(value, dict) and isinstance(out.get(key), dict)
                and key not in REPLACED_WHOLE):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path, seen=()) -> Dict[str, Any]:
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"circular 'base' chain through {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - SECTIONS
    if unknown:
        raise ValueError(f"{path.name}: unknown section(s) {sorted(unknown)}; "
                         f"allowed: {sorted(SECTIONS)}")
    if raw.get("base"):
        parent = _load_raw(path.parent / raw["base"], seen + (path,))
        parent.pop("name", None)
        parent.pop("description", None)
        raw = _merge(parent, {k: v for k, v in raw.items() if k != "base"})
    return raw


@dataclasses.dataclass
class Experiment:
    name: str
    description: str = ""
    from_run: Optional[str] = None
    schema: Dict[str, Any] = dataclasses.field(default_factory=dict)
    tokenizer: Dict[str, Any] = dataclasses.field(default_factory=dict)
    corpus: Dict[str, Any] = dataclasses.field(default_factory=dict)
    training: Dict[str, Any] = dataclasses.field(default_factory=dict)
    embeddings: Dict[str, Any] = dataclasses.field(default_factory=dict)
    xgboost: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, path, validate: bool = True) -> "Experiment":
        raw = _load_raw(Path(path))
        if not raw.get("name"):
            raise ValueError(f"{path}: 'name' is required (it names the run folder)")
        raw.pop("base", None)
        exp = cls(**{k: v for k, v in raw.items() if v is not None})
        if validate:
            exp.validate()
        return exp

    def to_yaml(self) -> str:
        """The fully resolved experiment (bases merged), as frozen in the run folder."""
        return yaml.safe_dump(dataclasses.asdict(self), sort_keys=False)

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        from src.tokenizer.bank_pipeline import BankSchema, BankTokenizerConfig

        for section, cls_ in (("tokenizer", BankTokenizerConfig), ("schema", BankSchema)):
            fields = {f.name for f in dataclasses.fields(cls_)}
            unknown = set(getattr(self, section)) - fields
            if unknown:
                raise ValueError(f"{self.name}: {section} has unknown key(s) {sorted(unknown)}")
        unknown = set(self.embeddings) - set(EMBEDDING_SETTINGS)
        if unknown:
            raise ValueError(f"{self.name}: embeddings has unknown key(s) {sorted(unknown)}; "
                             f"allowed: {sorted(EMBEDDING_SETTINGS)}")
        finetune = self.embeddings.get("finetune", "none")
        if finetune != "none":
            from src.finetune_spec import parse_finetune
            parse_finetune(finetune)                              # valid spec?
            if "learned" not in str(self.embeddings.get("pooling", "")):
                raise ValueError(f"{self.name}: finetune trains through the attention head, "
                                 "so pooling must include 'learned'")
        unknown = set(self.xgboost) - set(XGBOOST_SETTINGS) - {"params"}
        if unknown:
            raise ValueError(f"{self.name}: xgboost has unknown key(s) {sorted(unknown)}")
        if self.from_run and (self.tokenizer or self.training or self.schema or self.corpus):
            raise ValueError(
                f"{self.name}: from_run reuses {self.from_run}'s tokenizer and model, so "
                "tokenizer/corpus/training/schema settings would be ignored -- remove them"
            )
        unknown = set(self.corpus) - set(CORPUS_SETTINGS)
        if unknown:
            raise ValueError(f"{self.name}: corpus has unknown key(s) {sorted(unknown)}; "
                             f"allowed: {sorted(CORPUS_SETTINGS)}")
        if self.tokenizer:
            BankTokenizerConfig(**self.tokenizer).normalized()     # values valid?
            bad = [n for n in (self.tokenizer.get("extra_fields") or {})
                   if not (str(n).isascii() and str(n).isalnum())]
            if bad:
                raise ValueError(f"{self.name}: extra field names must be letters and digits "
                                 f"only, got {bad}")
        # The trainer creates unknown keys silently, so a typo ("max_step")
        # would leave the default in force. Check every key against the YAML.
        reference = yaml.safe_load(TRAINING_YAML.read_text())
        for key in self.training:
            node, parts = reference, key.split(".")
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
            if not (isinstance(node, dict) and (parts[-1] in node
                                                or key in OPTIONAL_TRAINING_KEYS)):
                raise ValueError(f"{self.name}: training key {key!r} is not in "
                                 f"{TRAINING_YAML.name}")
        # Sequences longer than the model's position table cannot be trained on.
        max_pos = self.training.get(
            "model.config.max_position_embeddings",
            reference["model"]["config"]["max_position_embeddings"])
        context = self.corpus.get("context_window")
        if context is not None and int(context) > int(max_pos):
            raise ValueError(f"{self.name}: corpus.context_window={context} exceeds "
                             f"model.config.max_position_embeddings={max_pos}")
        weights = self.training.get("loss_weights")
        if weights is not None and not isinstance(weights, dict):
            raise ValueError(f"{self.name}: loss_weights must be a mapping "
                             f"{{FIELD: weight}}, got {type(weights).__name__}")
        if weights:
            # Values here. Which fields exist depends on notebook 02's CONFIG
            # too (its own drop_fields / extra_fields), so the field names are
            # checked once, against the real tokenizer, at the start of notebook
            # 02 (check_loss_weights) -- still before any tokenizing.
            import math
            bad = [k for k, v in weights.items()
                   if isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v < 0]
            if bad:
                raise ValueError(f"{self.name}: loss_weights must be finite numbers >= 0; "
                                 f"invalid for {bad}")
        copies = self.corpus.get("chunk_copies")
        if copies is not None and (isinstance(copies, bool) or not isinstance(copies, int)
                                   or copies < 1):
            raise ValueError(f"{self.name}: corpus.chunk_copies must be an integer >= 1")
        if "pretrain_on_test" in self.corpus and not isinstance(self.corpus["pretrain_on_test"], bool):
            raise ValueError(f"{self.name}: corpus.pretrain_on_test must be true or false")
        if "logreg" in self.xgboost and not isinstance(self.xgboost["logreg"], bool):
            raise ValueError(f"{self.name}: xgboost.logreg must be true or false")
        if "model.architecture" in self.training:
            from src.architectures import ARCHITECTURES
            if self.training["model.architecture"] not in ARCHITECTURES:
                raise ValueError(f"{self.name}: unknown model.architecture "
                                 f"{self.training['model.architecture']!r}; "
                                 f"choose from {sorted(ARCHITECTURES)}")
        objective = str(self.training.get("model.objective", "clm"))
        if objective not in ("clm", "mlm", "nppr", "coles"):
            raise ValueError(f"{self.name}: model.objective must be clm, mlm, nppr or coles, "
                             f"got {objective!r}")
        from src.architectures import ENCODERS
        architecture = self.training.get(
            "model.architecture", yaml.safe_load(TRAINING_YAML.read_text())["model"].get("architecture"))
        encoder = architecture in ENCODERS
        if (objective == "mlm" and not encoder) or (objective in ("clm", "nppr") and encoder):
            raise ValueError(f"{self.name}: model.objective {objective} with model.architecture "
                             f"{architecture}: mlm needs an encoder ({sorted(ENCODERS)}), "
                             "clm and nppr a decoder (coles takes either)")
        if objective in ("nppr", "coles"):
            # Transfer-level objectives: the language-model head is not trained.
            if self.training.get("loss_weights"):
                raise ValueError(f"{self.name}: loss_weights is for objective clm only")
            if "surprise" in str(self.embeddings.get("pooling", "")):
                raise ValueError(f"{self.name}: pooling 'surprise' needs objective clm "
                                 "(it scores next-token predictions)")
            if encoder and self.embeddings.get("finetune", "none") != "none":
                raise ValueError(f"{self.name}: finetune is not supported with an encoder yet")
            alpha = self.training.get("nppr.alpha", 0.1)
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha < 1:
                raise ValueError(f"{self.name}: nppr.alpha must be in [0, 1) "
                                 "(0 = next transfer only)")
            self._check_number("nppr.decay_days", 60, lo=0.0)
            for key, default in (("nppr.past_events", 8), ("nppr.decoder_hidden", 512),
                                 ("coles.slices", 5), ("coles.min_txns", 8),
                                 ("coles.max_txns", 64), ("coles.neg_count", 5)):
                value = self.training.get(key, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{self.name}: {key} must be a positive integer")
            if self.training.get("coles.slices", 5) < 2:
                raise ValueError(f"{self.name}: coles.slices must be at least 2")
            if self.training.get("coles.min_txns", 8) > self.training.get("coles.max_txns", 64):
                raise ValueError(f"{self.name}: coles.min_txns must not exceed coles.max_txns")
            self._check_number("coles.margin", 0.5, lo=0.0)
        if objective == "mlm":
            if self.training.get("loss_weights"):
                raise ValueError(f"{self.name}: loss_weights is for objective clm only")
            if "surprise" in str(self.embeddings.get("pooling", "")):
                raise ValueError(f"{self.name}: pooling 'surprise' needs objective clm "
                                 "(it scores next-token predictions)")
            if self.embeddings.get("finetune", "none") != "none":
                raise ValueError(f"{self.name}: finetune is not supported with objective mlm yet")
            unit = self.training.get("mlm.mask_unit", "token")
            if unit not in ("token", "transaction"):
                raise ValueError(f"{self.name}: mlm.mask_unit must be token or transaction")
            prob = self.training.get("mlm.mask_prob", 0.15)
            if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not 0 < prob < 1:
                raise ValueError(f"{self.name}: mlm.mask_prob must be between 0 and 1")

    def _check_number(self, key: str, default, lo: float, hi: Optional[float] = None,
                      open_hi: bool = False) -> None:
        """``training[key]`` is a number above ``lo`` (and below ``hi``)."""
        value = self.training.get(key, default)
        ok = not isinstance(value, bool) and isinstance(value, (int, float)) and value > lo
        if ok and hi is not None:
            ok = value < hi if open_hi else value <= hi
        if not ok:
            bound = f" and below {hi}" if hi is not None else ""
            raise ValueError(f"{self.name}: {key} must be a number above {lo}{bound}")

    # -- locations ---------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return runs_root() / self.name

    @property
    def model_source_dir(self) -> Path:
        """Where the tokenizer, corpus and model live (another run with from_run)."""
        return runs_root() / self.from_run if self.from_run else self.run_dir

    def paths(self) -> Dict[str, Path]:
        """Notebook path variables, by name. Notebooks ``globals().update()`` this."""
        shared, run, src = shared_data_dir(), self.run_dir, self.model_source_dir
        tok, corpus, model = src / "tokenizer", src / "decoder_corpus", src / "model"
        return {
            "DATA_DIR": run,
            "SPLIT_DIR": shared / "temporal_split",
            "TOKENIZER_DIR": tok,
            "SCHEMA_PATH": tok / "schema.json",
            "STATE_PATH": tok / "tokenizer_state.json",
            "VOCAB_PATH": tok / "vocab.json",
            "CORPUS_DIR": corpus,
            "REPORT_PATH": corpus / "corpus_report.json",
            "TRAIN_CORPUS": corpus / "train_corpus.txt",
            "VAL_CORPUS": corpus / "val_corpus.txt",
            "MODEL_DIR": model,
            "CHECKPOINT_DIR": model / "checkpoints",
            "EMBED_DIR": run / "embeddings",
            "OUTPUTS_DIR": run / "outputs",
        }

    # -- settings ------------------------------------------------------------
    def tokenizer_config(self, config):
        """The notebook's CONFIG with this experiment's tokenizer settings applied."""
        return dataclasses.replace(config, **self.tokenizer).normalized()

    def apply_training(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Set this experiment's training keys on the loaded YAML config, in place.

        Values keep their YAML types (lists included), unlike the command-line
        ``apply_overrides`` path, which only parses scalars.
        """
        for key, value in self.training.items():
            node = cfg
            *parents, leaf = key.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = copy.deepcopy(value)
        return cfg

    def check_loss_weights(self, pipeline) -> None:
        """Every weighted field must be one *pipeline* emits (call once it is built)."""
        weights = self.training.get("loss_weights") or {}
        # The same field list the trainer reads from vocab.json.
        emitted = {p.upper() for p in pipeline.field_prefixes()}
        missing = sorted({str(k).upper() for k in weights} - emitted)
        if missing:
            raise ValueError(f"{self.name}: loss_weights names {missing}, which this tokenizer "
                             f"does not emit (dropped or disabled); it emits {sorted(emitted)}")

    def corpus_settings(self) -> Dict[str, Any]:
        return {CORPUS_SETTINGS[k]: v for k, v in self.corpus.items()}

    def embedding_settings(self) -> Dict[str, Any]:
        return {EMBEDDING_SETTINGS[k]: v for k, v in self.embeddings.items()}

    def xgboost_settings(self) -> Dict[str, Any]:
        return {XGBOOST_SETTINGS[k]: v for k, v in self.xgboost.items() if k != "params"}

    def xgboost_params(self) -> Dict[str, Any]:
        return dict(self.xgboost.get("params") or {})

    def summary_fields(self) -> Dict[str, Any]:
        """Flat ``section.key -> value`` of what this experiment changes."""
        out: Dict[str, Any] = {}
        for section in ("tokenizer", "corpus", "training", "embeddings", "xgboost"):
            for key, value in getattr(self, section).items():
                out[f"{section}.{key}"] = value
        if self.from_run:
            out["from_run"] = self.from_run
        return out


_ACTIVE: Dict[str, Experiment] = {}


def active_experiment() -> Optional[Experiment]:
    """The experiment named by ``LTM_EXPERIMENT``, or None (interactive use)."""
    path = os.environ.get(ENV_EXPERIMENT)
    if not path:
        return None
    if path not in _ACTIVE:
        _ACTIVE[path] = Experiment.load(path)
        print(f"Experiment  : {_ACTIVE[path].name}  ({path})")
    return _ACTIVE[path]
