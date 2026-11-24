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
Tokenizer pipeline for bank transfer / payment data (16-column schema).

Adapted from the card-transaction pipeline of NVIDIA's blueprint.  The
column set, the tokenizer chosen per column and the token order are all
declared in one place -- :class:`BankSchema` (names) and
:class:`BankTokenizerConfig` (vocabulary sizes) -- so adapting to a new
field is a one-line change.

Default token order (15 tokens per transaction, customer-scoped sequences)::

    ACCTH  TDIF  HOUR  DOW  MONTH  DC  SIGN  MVC  OPST  CMT  EMP  CTRY  BIC  CPTY  AMT

Context first (which account, when), then what kind of movement, then who --
coarse to fine, so the counterparty is predicted knowing its country and bank --
and the amount last, knowing all of it. The order only changes what each field
is conditioned on; the summed loss over a transaction stays comparable across
orders. ``BankTokenizerConfig.field_order`` sets it; ``None`` restores the
original order (AMT TDIF DC SIGN MVC CPTY BIC CTRY OPST CMT EMP HOUR DOW MONTH
ACCTH), which is also what a state file written before the option existed gets.

With ``sequence_scope="account"`` the ACCTH token drops out (the sequence is a
single account already), leaving 14. ``include_account_time_delta`` adds TDIFA
right after TDIF. ``account_token="order"`` replaces ACCTH with ACCT: the
account's number within its customer (see :func:`add_account_order`).

Column -> tokenizer mapping and the reasoning behind it:

======================  =========================  ===============================
column                  tokenizer                  why
======================  =========================  ===============================
amt_euro                QuantileBinTokenizer       continuous; quantile
                                                   bins fitted on train spread the
                                                   skewed EUR distribution evenly
datetime                TimeDeltaTokenizer         inter-transaction gap in log
                                                   bins, ~1.5x per bin over a
                                                   10-year horizon by default
                                                   (recency / velocity)
counterparty_account    HybridHashTokenizer        very many values; the counterparties
                                                   shared by the most customers get
                                                   their own token, the rest are
                                                   hashed to a fixed number of
                                                   buckets (like MERCH)
counterparty_bic        FrequencyVocabTokenizer    head keeps its
                                                   own token, tail -> OTHER
counterparty_country    FrequencyVocabTokenizer    country code
employee                FrequencyVocabTokenizer    ordinal tokens by
                                                   default (no names in vocab)
comment_structure       FrequencyVocabTokenizer    a few values
operations_status       FrequencyVocabTokenizer    status flag
movement_code           FrequencyVocabTokenizer    plays the MCC
                                                   role; optional coarse group
movement_sign           FrequencyVocabTokenizer    a few values
dc_flag                 FrequencyVocabTokenizer    boolean
description (optional)  TextTokenizer              free text -> text_slots tokens
                                                   (TXT), strategy chosen by
                                                   config.text_strategy; see text.py
month / dow / hour      FixedVocabTokenizer        bounded integer ranges
customer_id             (sequence key)             many values -- a token per
                                                   customer would blow up the
                                                   embedding table and memorize
                                                   identities; used for grouping
account_number          CategoricalHashTokenizer   sequence key in "account" scope;
                                                   in "customer" scope a customer's
                                                   accounts are interleaved, so a
                                                   hashed ACCTH token says which
                                                   account a transaction is on;
                                                   or ACCT (account_token="order"):
                                                   1 = the customer's first account
======================  =========================  ===============================

A customer having several accounts is the normal case, and
``BankTokenizerConfig.sequence_scope`` decides what that means:

``"customer"`` (default)
    One interleaved, chronological sequence per customer, with an ACCTH token on
    every transaction. Longer sequences, cross-account behaviour (own-account
    transfers, activity spread over accounts) and history for a newly opened
    account.
``"account"``
    One sequence per (customer, account), like TabFormer's (user, card). Cleaner
    if a customer's accounts serve genuinely different purposes.

Either way transactions are never mixed across customers, and
``BankTransferTokenizerPipeline.group_cols`` is the single place the choice takes
effect -- sorting, gaps, corpus chunking and history windows all read it.

Fitted steps (amount bins, frequency vocabularies) MUST be fitted on the
training split only and then reloaded via
:meth:`TokenizerPipeline.save` / :meth:`BankTransferTokenizerPipeline.load`.
Re-fitting per split would silently assign different token IDs to the same
value in val/test.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import cudf
import numpy as np

from ._hash import HASH_ALGO as _HASH_ALGO
from ._hash import stable_hash
from .categorical_hash import CategoricalHashTokenizer
from .field_kinds import get_field_kind
from .fixed_vocab import FixedVocabTokenizer
from .frequency_vocab import FrequencyVocabTokenizer
from .hybrid_hash import HybridHashTokenizer
from .pipeline import TokenizerPipeline
from .quantile_numeric import QuantileBinTokenizer
from .text import TextTokenizer
from .tail import known_tail_strategies
from .text import known_text_strategies
from .timedelta import TimeDeltaTokenizer

# ── Column-name handling ──────────────────────────────────────────────

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    """``"Counterparty BIC"`` -> ``"counterparty_bic"``."""
    return _NON_ALNUM.sub("_", str(name).strip().lower()).strip("_")


#: Common source-column spellings mapped onto the canonical schema names.
#: Applied only when the canonical column is not already present.
DEFAULT_ALIASES: Dict[str, str] = {
    "cust_id": "customer_id",
    "customer": "customer_id",
    "customer_nr": "customer_id",
    "client_id": "customer_id",
    "account": "account_number",
    "account_no": "account_number",
    "account_nr": "account_number",
    "acct": "account_number",
    "iban": "account_number",
    "date_time": "datetime",
    "timestamp": "datetime",
    "ts": "datetime",
    "booking_datetime": "datetime",
    "cp_account": "counterparty_account",
    "counterpart_account": "counterparty_account",
    "counterparty_iban": "counterparty_account",
    "bic": "counterparty_bic",
    "cp_bic": "counterparty_bic",
    "country": "counterparty_country",
    "cp_country": "counterparty_country",
    "operation_status": "operations_status",
    "op_status": "operations_status",
    "mvt_code": "movement_code",
    "movement_cd": "movement_code",
    "sign": "movement_sign",
    "mvt_sign": "movement_sign",
    "dc": "dc_flag",
    "debit_credit": "dc_flag",
    "dc_ind": "dc_flag",
    "day_of_week": "dow",
    "weekday": "dow",
    "amount": "amt_euro",
    "amt_eur": "amt_euro",
    "amount_eur": "amt_euro",
    "amount_euro": "amt_euro",
}


@dataclass
class BankSchema:
    """Source column names for the 16-column transfer schema (+ label).

    Override any field if your extract uses a different name, or pass
    ``aliases={"source_col": "canonical_col"}`` for bulk renames.  Names are
    matched after normalization (lowercase, non-alphanumerics -> ``_``), so
    ``"Counterparty BIC"`` already resolves to ``counterparty_bic``.
    """

    customer_id: str = "customer_id"
    account_number: str = "account_number"
    datetime: str = "datetime"
    counterparty_account: str = "counterparty_account"
    counterparty_bic: str = "counterparty_bic"
    counterparty_country: str = "counterparty_country"
    employee: str = "employee"
    comment_structure: str = "comment_structure"
    operations_status: str = "operations_status"
    movement_code: str = "movement_code"
    movement_sign: str = "movement_sign"
    dc_flag: str = "dc_flag"
    month: str = "month"
    dow: str = "dow"
    hour: str = "hour"
    amount: str = "amt_euro"
    #: Free-text description. Optional: only read when
    #: ``BankTokenizerConfig.text_strategy`` is set.
    description: Optional[str] = None
    label: Optional[str] = "label"
    aliases: Dict[str, str] = field(default_factory=dict)

    #: Columns that must exist for tokenization (label excluded).
    REQUIRED_FIELDS: ClassVar[Tuple[str, ...]] = (
        "customer_id", "account_number", "datetime", "counterparty_account",
        "counterparty_bic", "counterparty_country", "employee",
        "comment_structure", "operations_status", "movement_code",
        "movement_sign", "dc_flag", "amount",
    )

    @property
    def feature_columns(self) -> List[str]:
        """The 16 raw feature columns, e.g. for the XGBoost baseline."""
        return [
            self.customer_id, self.account_number, self.datetime,
            self.counterparty_account, self.counterparty_bic,
            self.counterparty_country, self.employee, self.comment_structure,
            self.operations_status, self.movement_code, self.movement_sign,
            self.dc_flag, self.month, self.dow, self.hour, self.amount,
        ]

    @property
    def customer_cols(self) -> List[str]:
        """Relationship-level key: all accounts of one customer."""
        return [self.customer_id]

    @property
    def account_cols(self) -> List[str]:
        """Account-level key: one customer account."""
        return [self.customer_id, self.account_number]

    def rename_map(self, columns: Sequence[str]) -> Dict[str, str]:
        """Build ``{present_column: canonical_column}`` for a DataFrame."""
        present = {normalize_name(c): c for c in columns}
        canonical = {getattr(self, f) for f in self.REQUIRED_FIELDS}
        canonical |= {self.month, self.dow, self.hour}
        if self.description:
            canonical.add(self.description)
        if self.label:
            canonical.add(self.label)

        table = dict(DEFAULT_ALIASES)
        table.update({normalize_name(k): v for k, v in self.aliases.items()})

        mapping: Dict[str, str] = {}
        for norm, original in present.items():
            target = norm if norm in canonical else table.get(norm)
            if target and target not in present and target != original:
                mapping[original] = target
            elif target and norm != original:
                # `target and` matters: only columns the schema knows about get
                # canonicalized. Without it every column was normalized, and
                # normalize_name() strips leading/trailing underscores --
                # silently turning the bookkeeping columns `__row_id__` into
                # `row_id` and `_target` into `target`. preprocess() promises to
                # keep every original column, and notebooks 04/05 look those two
                # up by name, so renaming them broke row alignment and labels.
                mapping[original] = norm
        return mapping

    def missing(self, columns: Sequence[str]) -> List[str]:
        cols = set(columns)
        return [
            getattr(self, f) for f in self.REQUIRED_FIELDS
            if getattr(self, f) not in cols
        ]


#: Default emission order of the tokenizer steps, by token prefix. Steps that a
#: configuration does not enable (TDIFA, MVGRP, ACCTH in account scope, CUSTH)
#: are skipped. See the module docstring for the reasoning.
DEFAULT_FIELD_ORDER: Tuple[str, ...] = (
    "ACCTH", "ACCT", "CUSTH", "TDIF", "TDIFA", "HOUR", "DOW", "MONTH",
    "DC", "SIGN", "MVC", "MVGRP", "OPST", "CMT", "EMP", "TXT",
    "CTRY", "BIC", "CPTY", "AMTREL", "AMT",
)


@dataclass
class BankTokenizerConfig:
    """Vocabulary sizes and derived-feature switches.

    Vocabulary caps are upper bounds: a frequency vocabulary that sees fewer
    distinct values stays smaller, so the final ``vocab_size`` is only known
    after :meth:`fit`.
    """

    #: What one sequence is.
    #:   "customer" -- all accounts of a customer interleaved chronologically.
    #:     Longer sequences, cross-account behaviour (own-account transfers,
    #:     activity spreading over accounts) and context for a newly opened
    #:     account. An account-identity token is added so the model can tell
    #:     which account each transaction belongs to.
    #:   "account"  -- one sequence per (customer, account), like TabFormer's
    #:     (user, card). Cleaner behavioural unit if a customer's accounts serve
    #:     genuinely different purposes.
    sequence_scope: str = "customer"
    # amount
    #: AMTREL token: log-amount minus the mean log-amount of the sequence's
    #: earlier transactions ("unusual for this customer"), quantile-binned;
    #: AMTREL_FIRST when there is no earlier transaction in the frame. Causal --
    #: only the past is used. In val/test frames the past starts at the split
    #: cutoff (plus notebook 04's history context), as for TDIF.
    include_relative_amount: bool = False
    relative_amount_bins: int = 16
    amount_bins: int = 32
    amount_signed_log: bool = False
    amount_clip_quantiles: Optional[Tuple[float, float]] = (0.0005, 0.9995)
    # inter-transaction time
    include_time_delta: bool = True
    #: Bins spread over the log of the horizon, so resolution is multiplicative:
    #: step = horizon ** (1 / bins). 48 bins over 10 years is ~1.50x per bin
    #: (32 would be 1.84x). Bins cost one vocabulary row each -- 16 extra rows in
    #: an ~8k vocabulary -- so buy resolution here rather than shrinking the
    #: horizon.
    time_delta_bins: int = 48
    #: Upper end of the gap range, in years; longer gaps clip into the top bin.
    #: Set it to the span of your extract so nothing meaningful clips. Because
    #: the scale is logarithmic, a longer horizon is nearly free: at a fixed
    #: 1.5x resolution, going from a 1-year to a 10-year horizon costs 6 bins.
    #: Shrinking it does NOT buy much -- 2 years over 32 bins (1.75x) is barely
    #: finer than 10 years over 32 (1.84x), and it makes every gap above ~8
    #: months the same token. Notebook 02 prints the gap distribution against
    #: this setting so you can trim the unused tail on real data.
    time_delta_max_years: float = 10.0
    #: Adds a second gap token (TDIFA) measured within the account, which
    #: customer-scoped sequences otherwise lose. Costs one token per transaction.
    include_account_time_delta: bool = False
    #: Map gap interval i to TDIF_i. Off, TDIF_0 is never emitted and gaps at
    #: or beyond the horizon share the last real bin's token.
    time_delta_zero_based_bins: bool = True
    #: Emit TDIF_FIRST (and TDIFA_FIRST) when there is no previous transaction,
    #: instead of filling the gap with 0 -- which made "first transaction" and
    #: "same second as the previous one" the same token.
    time_delta_first_token: bool = True
    # counterparty
    counterparty_hash_size: int = 4096
    #: Counterparties used by the most distinct customers in train get a
    #: dedicated CPTY_T<rank> token; the rest are hashed as before. 0 -> hash
    #: everything (the original behaviour).
    counterparty_top_k: int = 2048
    #: Floor for a dedicated token: at least this many distinct customers.
    counterparty_min_customers: int = 10
    bic_max_vocab: int = 2048
    country_max_vocab: int = 256
    # operational fields
    employee_max_vocab: int = 128
    employee_readable: bool = False  # ordinal tokens: no employee names in vocab
    comment_structure_max_vocab: int = 16
    operations_status_max_vocab: int = 4
    movement_code_max_vocab: int = 512
    movement_sign_max_vocab: int = 8
    dc_flag_max_vocab: int = 4
    #: If set, adds a coarse MVGRP token derived from ``movement_code // divisor``
    #: (the analogue of TabFormer's MCC -> industry CAT token).  Leave None
    #: unless your movement codes are hierarchical.
    movement_group_divisor: Optional[int] = None
    movement_group_max_vocab: int = 64
    # Identity tokens. Raw IDs never get their own token (tens of thousands of
    # each), but a hashed bucket is cheap and split-stable.
    #: Account identity token. None -> on for customer-scoped sequences (where
    #: it disambiguates interleaved accounts), off for account-scoped ones
    #: (where the whole sequence is one account already).
    include_account_hash: Optional[bool] = None
    account_hash_size: int = 1024
    #: How that account token is made (when include_account_hash is on).
    #:   "hash"  -- ACCTH_<bucket>: a hash of the account number. A bucket mixes
    #:     unrelated accounts of many customers, so its meaning is noise.
    #:   "order" -- ACCT_<n>: the account's number within its customer, in order
    #:     of first appearance (ACCT_01 = the customer's first account). Same
    #:     meaning for every customer, no collisions, ~16 tokens. Reads the
    #:     ``account_order`` column, computed once over the full history (notebook
    #:     01, or scripts/add_account_order.py for existing splits).
    account_token: str = "hash"
    #: Last ACCT number: accounts from this one on share ACCT_<max>.
    account_order_max: int = 16
    include_customer_hash: bool = False
    customer_hash_size: int = 1024
    # temporal ranges (change if your extract encodes dow as 1-7 etc.)
    hour_range: Tuple[int, int] = (0, 23)
    dow_range: Tuple[int, int] = (0, 6)
    month_range: Tuple[int, int] = (1, 12)
    # preprocessing
    derive_time_parts: bool = True  # recompute hour/dow/month from datetime
    datetime_format: Optional[str] = None  # None -> let cuDF infer
    sort_transactions: bool = True
    #: Emission order of the steps, by token prefix (see DEFAULT_FIELD_ORDER).
    #: None -> the original order.
    field_order: Optional[Tuple[str, ...]] = DEFAULT_FIELD_ORDER
    #: Fields to leave out of the token stream, by token prefix (e.g. ("EMP",)).
    #: The source column stays required and is still read; it just emits no
    #: token. tokens_per_transaction, chunk size and the per-field diagnostics
    #: all follow from the remaining steps.
    drop_fields: Tuple[str, ...] = ()
    # free text (schema.description) -- see src/tokenizer/text.py
    #: None -> no text field. Else the strategy that turns the text into
    #: ``text_slots`` tokens: "words", "bag", "category" or "hashed"
    #: (known_text_strategies() lists the registered ones).
    text_strategy: Optional[str] = None
    text_slots: int = 4
    text_max_vocab: int = 4096
    #: A word enters the vocabulary only if this many distinct customers use it
    #: (keeps names and personal references out).
    text_min_customers: int = 5
    text_hash_size: int = 2048     # "hashed" strategy only
    text_readable: bool = False    # True -> TXT_<word>; False -> TXT_<rank>
    # values without a token of their own -- see src/tokenizer/tail.py
    #: Applies to every frequency-vocabulary field (BIC, CTRY, MVC, ...).
    #: "rare_bins": rarity bands, "hash": hash buckets, "other": one OTHER
    #: token (the original; its embedding is never trained when the cap
    #: exceeds the field's cardinality).
    tail_strategy: str = "rare_bins"
    #: Values seen fewer times than this in train go to the tail -- which is
    #: what trains the tail tokens that unseen values are mapped to.
    tail_min_count: int = 10
    tail_buckets: int = 16         # "hash" tail only
    #: New fields, by token prefix -> {"column": ..., "kind": ..., options}.
    #: Kinds: categorical | identifier | numeric | text (field_kinds.py).
    extra_fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    #: Values that reproduce the behaviour from before each option existed.
    #: ``from_state`` applies them to keys a saved state does not have, so an
    #: old tokenizer -- and the checkpoint trained on it -- reloads unchanged
    #: instead of silently picking up the new defaults.
    LEGACY_DEFAULTS: ClassVar[Dict[str, Any]] = {
        "time_delta_zero_based_bins": False,
        "time_delta_first_token": False,
        "counterparty_top_k": 0,
        "field_order": None,
        "tail_strategy": "other",
        "tail_min_count": 1,
    }

    def normalized(self) -> "BankTokenizerConfig":
        """Validate, resolve scope-dependent defaults, restore tuples from JSON."""
        if self.sequence_scope not in ("customer", "account"):
            raise ValueError(
                f"sequence_scope must be 'customer' or 'account', "
                f"got {self.sequence_scope!r}"
            )
        if self.include_account_hash is None:
            self.include_account_hash = self.sequence_scope == "customer"
        if self.account_token not in ("hash", "order"):
            raise ValueError(f"account_token must be 'hash' or 'order', got {self.account_token!r}")
        if self.account_order_max < 1:
            raise ValueError(f"account_order_max must be >= 1, got {self.account_order_max}")
        if self.amount_clip_quantiles is not None:
            self.amount_clip_quantiles = tuple(self.amount_clip_quantiles)
        self.hour_range = tuple(self.hour_range)
        self.dow_range = tuple(self.dow_range)
        self.month_range = tuple(self.month_range)
        if self.field_order is not None:
            self.field_order = tuple(str(name).upper() for name in self.field_order)
        self.drop_fields = tuple(str(name).upper() for name in self.drop_fields)
        # Strategy names are checked here, not when the step is first built
        # mid-tokenization, so a typo in an experiment fails before any work.
        if self.tail_strategy not in known_tail_strategies():
            raise ValueError(f"tail_strategy={self.tail_strategy!r}; choose from "
                             f"{known_tail_strategies()}")
        if self.text_strategy and self.text_strategy not in known_text_strategies():
            raise ValueError(f"text_strategy={self.text_strategy!r}; choose from "
                             f"{known_text_strategies()}")
        self.extra_fields = {str(k).upper(): dict(v) for k, v in self.extra_fields.items()}
        for name, spec in self.extra_fields.items():
            if name in DEFAULT_FIELD_ORDER:
                raise ValueError(f"extra field {name!r} clashes with a built-in field")
            if "column" not in spec or "kind" not in spec:
                raise ValueError(f"extra field {name!r} needs 'column' and 'kind'")
            get_field_kind(spec["kind"]).resolve(spec)   # validates the options
        return self


# ── Derived column names produced by preprocess() ─────────────────────

DT_COL = "_dt"
AMT_COL = "amt_val"
TDIF_COL = "time_delta_s"
TDIFA_COL = "time_delta_acct_s"
CPTY_COL = "cpty_hash"
BIC_COL = "bic_clean"
CTRY_COL = "ctry_clean"
EMP_COL = "emp_clean"
CMT_COL = "cmt_val"
OPST_COL = "opst_clean"
MVC_COL = "mvc_val"
MVGRP_COL = "mvgrp_val"
SIGN_COL = "sign_clean"
DC_COL = "dc_clean"
HOUR_COL = "hour_val"
DOW_COL = "dow_val"
MONTH_COL = "month_val"
CUSTH_COL = "cust_hash"
ACCTH_COL = "acct_hash"
ACCT_COL = "acct_order_val"
#: Source column written by add_account_order() (notebook 01).
ACCOUNT_ORDER_COL = "account_order"
TXT_COL = "txt_raw"
AMTREL_COL = "amt_rel"


class BankTransferTokenizerPipeline(TokenizerPipeline):
    """Ready-to-use tokenizer pipeline for the bank transfer schema.

    Parameters
    ----------
    schema : BankSchema
        Source column names.
    config : BankTokenizerConfig
        Vocabulary sizes and derived-feature switches.
    """

    #: ``preprocess()`` builds the CPTY / CUSTH / ACCTH hash-bucket columns with
    #: ``_hash.stable_hash``. Recorded in the state file so a corpus fitted
    #: under a different hash cannot be reused silently.
    HASH_ALGO: ClassVar[Optional[str]] = _HASH_ALGO

    def __init__(
        self,
        schema: Optional[BankSchema] = None,
        config: Optional[BankTokenizerConfig] = None,
        **kwargs,
    ):
        # Multi-stream fit/transform is off by default: cuDF operations run on
        # their own stream regardless of the CuPy stream context, so a step that
        # mixes cuDF and CuPy (quantile bins, time deltas) would consume data
        # across streams without an event sync. The streams buy little here and
        # correctness is not worth trading for it -- pass use_streams=True to
        # opt back in.
        kwargs.setdefault("use_streams", False)
        super().__init__(**kwargs)
        self.schema = schema or BankSchema()
        self.config = (config or BankTokenizerConfig()).normalized()
        self._configure_steps()

    # ------------------------------------------------------------------
    # Step configuration
    # ------------------------------------------------------------------

    def _configure_steps(self) -> None:
        cfg = self.config
        s = self.schema

        # Steps keyed by token prefix, in the original emission order -- which
        # is what field_order=None reproduces. Disabled steps are absent.
        fields: Dict[str, Tuple[Any, Any]] = {}

        # Amount — quantile bins fitted on the training split
        fields["AMT"] = (AMT_COL, QuantileBinTokenizer(
            prefix="AMT",
            num_bins=cfg.amount_bins,
            signed_log=cfg.amount_signed_log,
            clip_quantiles=cfg.amount_clip_quantiles,
        ))

        # Amount relative to the sequence's own past (fitted on train)
        if cfg.include_relative_amount:
            fields["AMTREL"] = (AMTREL_COL, QuantileBinTokenizer(
                prefix="AMTREL", num_bins=cfg.relative_amount_bins,
                clip_quantiles=(0.001, 0.999), missing_label="FIRST",
            ))

        # Inter-transaction gap — log-spaced bins, measured along the sequence
        if cfg.include_time_delta:
            fields["TDIF"] = (TDIF_COL, TimeDeltaTokenizer(
                num_bins=cfg.time_delta_bins, special_token="TDIF",
                max_years=cfg.time_delta_max_years,
                zero_based_bins=cfg.time_delta_zero_based_bins,
                first_token=cfg.time_delta_first_token,
            ))
        # Gap within the account (only meaningful when the sequence
        # interleaves several accounts)
        if cfg.include_account_time_delta:
            fields["TDIFA"] = (TDIFA_COL, TimeDeltaTokenizer(
                num_bins=cfg.time_delta_bins, special_token="TDIFA",
                max_years=cfg.time_delta_max_years,
                zero_based_bins=cfg.time_delta_zero_based_bins,
                first_token=cfg.time_delta_first_token,
            ))

        # Direction and sign
        fields["DC"] = (DC_COL, self._freq(
            prefix="DC", max_vocab=cfg.dc_flag_max_vocab,
        ))
        fields["SIGN"] = (SIGN_COL, self._freq(
            prefix="SIGN", max_vocab=cfg.movement_sign_max_vocab,
        ))

        # Movement code (MCC analogue) + optional coarse group
        fields["MVC"] = (MVC_COL, self._freq(
            prefix="MVC", max_vocab=cfg.movement_code_max_vocab,
        ))
        if cfg.movement_group_divisor:
            fields["MVGRP"] = (MVGRP_COL, self._freq(
                prefix="MVGRP", max_vocab=cfg.movement_group_max_vocab,
            ))

        # Counterparty identity / bank / country. With counterparty_top_k > 0 the
        # step also reads the customer column, to rank counterparties by how
        # many distinct customers use them (fit time only).
        if cfg.counterparty_top_k > 0:
            fields["CPTY"] = ([CPTY_COL, s.customer_id], HybridHashTokenizer(
                vocab_limit=cfg.counterparty_hash_size, special_token="CPTY",
                top_k=cfg.counterparty_top_k,
                min_customers=cfg.counterparty_min_customers,
            ))
        else:
            fields["CPTY"] = (CPTY_COL, CategoricalHashTokenizer(
                vocab_limit=cfg.counterparty_hash_size, special_token="CPTY",
            ))
        fields["BIC"] = (BIC_COL, self._freq(
            prefix="BIC", max_vocab=cfg.bic_max_vocab,
        ))
        fields["CTRY"] = (CTRY_COL, self._freq(
            prefix="CTRY", max_vocab=cfg.country_max_vocab,
        ))

        # Operational context
        fields["OPST"] = (OPST_COL, self._freq(
            prefix="OPST", max_vocab=cfg.operations_status_max_vocab,
        ))
        fields["CMT"] = (CMT_COL, self._freq(
            prefix="CMT", max_vocab=cfg.comment_structure_max_vocab,
        ))
        fields["EMP"] = (EMP_COL, self._freq(
            prefix="EMP",
            max_vocab=cfg.employee_max_vocab,
            readable=cfg.employee_readable,
        ))

        # Free text -> text_slots tokens. Reads the customer column at fit time
        # to count distinct customers per word.
        if cfg.text_strategy:
            fields["TXT"] = ([TXT_COL, s.customer_id], TextTokenizer(
                prefix="TXT",
                strategy=cfg.text_strategy,
                slots=cfg.text_slots,
                max_vocab=cfg.text_max_vocab,
                min_customers=cfg.text_min_customers,
                hash_size=cfg.text_hash_size,
                readable=cfg.text_readable,
            ))

        # Calendar position. zero_based_index=True keeps local indices dense
        # from 0 whatever min_val is (MONTH starts at 1, and dow may start at 1
        # in some extracts), so a step never overruns its vocabulary slot.
        fields["HOUR"] = (HOUR_COL, FixedVocabTokenizer(
            prefix="HOUR", min_val=cfg.hour_range[0], max_val=cfg.hour_range[1],
            pad_width=2, zero_based_index=True,
        ))
        fields["DOW"] = (DOW_COL, FixedVocabTokenizer(
            prefix="DOW", min_val=cfg.dow_range[0], max_val=cfg.dow_range[1],
            zero_based_index=True,
        ))
        fields["MONTH"] = (MONTH_COL, FixedVocabTokenizer(
            prefix="MONTH", min_val=cfg.month_range[0], max_val=cfg.month_range[1],
            pad_width=2, zero_based_index=True,
        ))

        # Which account this transaction belongs to: a hash bucket (ACCTH), or
        # the account's number within its customer (ACCT). The number is not
        # computed here: ranking within each frame would shift between splits
        # and windows whenever an account has no transactions in one of them.
        # It comes from the account_order column, computed once over the full
        # history (add_account_order).
        if cfg.include_account_hash and cfg.account_token == "order":
            fields["ACCT"] = (ACCT_COL, FixedVocabTokenizer(
                prefix="ACCT", min_val=1, max_val=cfg.account_order_max,
                pad_width=2, zero_based_index=True,
            ))
        elif cfg.include_account_hash:
            fields["ACCTH"] = (ACCTH_COL, CategoricalHashTokenizer(
                vocab_limit=cfg.account_hash_size, special_token="ACCTH",
            ))
        if cfg.include_customer_hash:
            fields["CUSTH"] = (CUSTH_COL, CategoricalHashTokenizer(
                vocab_limit=cfg.customer_hash_size, special_token="CUSTH",
            ))

        # New fields declared by kind (field_kinds.py).
        for name, spec in cfg.extra_fields.items():
            kind = get_field_kind(spec["kind"])
            column = self._extra_column(name)
            cols = [column, s.customer_id] if kind.needs_customer else column
            fields[name] = (cols, kind.tokenizer(name, kind.resolve(spec), self))

        known = set(DEFAULT_FIELD_ORDER) | set(cfg.extra_fields)
        unknown = [name for name in cfg.drop_fields if name not in known]
        if unknown:
            raise ValueError(
                f"drop_fields has unknown fields {unknown}; "
                f"choose from {sorted(known)}"
            )
        for name in cfg.drop_fields:
            fields.pop(name, None)
        if not fields:
            raise ValueError("drop_fields removes every field; nothing to tokenize")

        for name in self._emission_order(fields):
            column, tokenizer = fields[name]
            self.add_step(column, tokenizer)

    def fit(self, df):
        """Fit on the training split -- where rules for *new* tokenizers apply.

        Loading a saved state never calls this, so a rule tightened later
        never stops an already-fitted tokenizer from loading.
        """
        for name in self.config.extra_fields:
            # Tokens are "<NAME>_<value>": a name with "_" would make a token's
            # field ambiguous (CPTY_TYPE_3: CPTY or CPTY_TYPE?).
            if not (name.isascii() and name.isalnum()):
                raise ValueError(f"extra field name {name!r} must be letters and digits only")
        return super().fit(df)

    def _freq(self, prefix: str, max_vocab: int, **kwargs) -> FrequencyVocabTokenizer:
        """A frequency-vocabulary step with the configured tail strategy."""
        cfg = self.config
        return FrequencyVocabTokenizer(
            prefix=prefix, max_vocab=max_vocab, min_count=cfg.tail_min_count,
            tail=cfg.tail_strategy, tail_buckets=cfg.tail_buckets, **kwargs,
        )

    @staticmethod
    def _extra_column(name: str) -> str:
        return f"x_{name.lower()}"

    def _emission_order(self, fields: Dict[str, Any]) -> List[str]:
        """Resolve ``config.field_order`` against the steps actually enabled."""
        order = self.config.field_order
        if order is None:
            return list(fields)
        known = set(DEFAULT_FIELD_ORDER) | set(self.config.extra_fields)
        unknown = [name for name in order if name not in known]
        if unknown:
            raise ValueError(
                f"field_order has unknown fields {unknown}; "
                f"choose from {list(DEFAULT_FIELD_ORDER)}"
            )
        if len(set(order)) != len(order):
            raise ValueError(f"field_order lists a field twice: {list(order)}")
        # Extra fields not placed explicitly go just before the amount, which
        # stays last (predicted knowing everything else).
        unplaced = [n for n in self.config.extra_fields if n in fields and n not in order]
        if unplaced:
            at = order.index("AMT") if "AMT" in order else len(order)
            order = tuple(order[:at]) + tuple(unplaced) + tuple(order[at:])
        missing = [name for name in fields if name not in order]
        if missing:
            raise ValueError(
                f"field_order does not place the enabled field(s) {missing}. "
                "List every enabled field (disabled ones are skipped)."
            )
        return [name for name in order if name in fields]

    # ------------------------------------------------------------------
    # Sequence key
    # ------------------------------------------------------------------

    @property
    def group_cols(self) -> List[str]:
        """Columns that define one sequence, per ``config.sequence_scope``.

        A customer with several accounts therefore yields either one
        interleaved sequence (``"customer"``) or one sequence per account
        (``"account"``).  Everything downstream -- sorting, the inter-transaction
        gap, corpus chunking, history windows -- reads this property, so the
        scope only has to be set once.
        """
        if self.config.sequence_scope == "customer":
            return self.schema.customer_cols
        return self.schema.account_cols

    # ------------------------------------------------------------------
    # Preprocessing  — raw DataFrame -> pipeline-ready columns
    # ------------------------------------------------------------------

    def preprocess(self, df: cudf.DataFrame, sort: Optional[bool] = None) -> cudf.DataFrame:
        """Normalize names, derive the columns the steps consume, sort by time.

        Returns a new DataFrame that keeps every original column (labels and
        raw features travel with the frame) plus the derived ones.  Rows are
        sorted by ``(*group_cols, datetime)`` -- required for the time-delta
        feature, corpus chunking and history windows -- so carry an explicit
        row-id column if you need to restore input order.
        """
        s, cfg = self.schema, self.config
        sort = cfg.sort_transactions if sort is None else sort

        df = df.rename(columns=s.rename_map(list(df.columns)))
        missing = s.missing(list(df.columns))
        if missing:
            raise ValueError(
                f"Missing required columns after renaming: {missing}\n"
                f"Available: {sorted(df.columns)}\n"
                "Set the names on BankSchema(...) or pass "
                "BankSchema(aliases={'source_name': 'canonical_name'})."
            )

        if not cfg.derive_time_parts:
            time_cols = [s.month, s.dow, s.hour]
            absent = [c for c in time_cols if c not in df.columns]
            if absent:
                raise ValueError(
                    f"derive_time_parts=False but {absent} are missing. Either "
                    "provide those columns or keep derive_time_parts=True to "
                    f"compute them from '{s.datetime}'."
                )

        # -- timestamp -------------------------------------------------
        dt = df[s.datetime]
        if dt.dtype.kind != "M":
            try:
                dt = cudf.to_datetime(dt, format=cfg.datetime_format)
            except Exception as exc:
                raise ValueError(
                    f"Could not parse '{s.datetime}' as a timestamp. Pass "
                    "BankTokenizerConfig(datetime_format='%Y-%m-%d %H:%M:%S'). "
                    f"Original error: {exc}"
                ) from exc
        df[DT_COL] = dt

        if sort:
            df = df.sort_values(self.group_cols + [DT_COL]).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        # -- calendar position ----------------------------------------
        if cfg.derive_time_parts:
            df[HOUR_COL] = df[DT_COL].dt.hour.astype("int32")
            df[DOW_COL] = df[DT_COL].dt.dayofweek.astype("int32")
            df[MONTH_COL] = df[DT_COL].dt.month.astype("int32")
        else:
            df[HOUR_COL] = df[s.hour].fillna(0).astype("int32")
            df[DOW_COL] = df[s.dow].fillna(0).astype("int32")
            df[MONTH_COL] = df[s.month].fillna(1).astype("int32")

        # -- amount ----------------------------------------------------
        df[AMT_COL] = self._to_float(df[s.amount])
        if cfg.include_relative_amount:
            # Rows are sorted by (sequence, time), so a cumulative sum minus the
            # row itself is the sum over the sequence's earlier transactions.
            df["_log_amt"] = np.log1p(df[AMT_COL].abs())   # ufunc: pandas and cuDF
            grouped = df.groupby(self.group_cols)
            past_sum = grouped["_log_amt"].cumsum() - df["_log_amt"]
            past_n = grouped.cumcount()
            rel = df["_log_amt"] - past_sum / past_n.where(past_n > 0)
            df[AMTREL_COL] = rel.where(past_n > 0)          # missing -> AMTREL_FIRST
            df = df.drop(columns=["_log_amt"])

        # -- inter-transaction gaps (seconds) ---------------------------------
        # Along the sequence (customer or account, per sequence_scope), and
        # optionally within the account as well. The first transaction of a
        # series has no gap: it stays missing when the tokenizer has a FIRST
        # token for it, and is filled with 0 otherwise (the original behaviour).
        # "First" means first in this frame -- for val/test that is the first
        # transaction after the split cutoff, not the customer's first ever.
        if cfg.include_time_delta:
            delta = df.groupby(self.group_cols)[DT_COL].diff()
            df[TDIF_COL] = self._gap_seconds(delta)
        if cfg.include_account_time_delta:
            delta_acct = df.groupby(s.account_cols)[DT_COL].diff()
            df[TDIFA_COL] = self._gap_seconds(delta_acct)

        # -- counterparty ---------------------------------------------
        # stable_hash, not cudf's hash_values(): the bucket index becomes part
        # of the token (CPTY_<n>), so it must not depend on the dataframe
        # library or its version. See src/tokenizer/_hash.py.
        df[CPTY_COL] = stable_hash(self._clean_str(df[s.counterparty_account]))
        df[BIC_COL] = self._clean_str(df[s.counterparty_bic])
        df[CTRY_COL] = self._clean_str(df[s.counterparty_country])

        # -- operational fields ---------------------------------------
        df[EMP_COL] = self._clean_str(df[s.employee])
        df[OPST_COL] = self._clean_str(df[s.operations_status])
        df[SIGN_COL] = self._clean_str(df[s.movement_sign])
        df[DC_COL] = self._clean_str(df[s.dc_flag])
        df[CMT_COL] = df[s.comment_structure].fillna(-1).astype("int64")

        mvc = df[s.movement_code].fillna(-1).astype("int64")
        df[MVC_COL] = mvc
        if cfg.movement_group_divisor:
            df[MVGRP_COL] = mvc // int(cfg.movement_group_divisor)

        # -- free text: raw; the text step cleans it (strategy-specific) --
        if cfg.text_strategy:
            if not s.description or s.description not in df.columns:
                raise ValueError(
                    f"text_strategy={cfg.text_strategy!r} needs the description "
                    f"column, but BankSchema.description={s.description!r} is "
                    "not in the data. Set it in notebook 01's SCHEMA."
                )
            df[TXT_COL] = df[s.description].fillna("").astype("str")

        # -- extra fields, cleaned by their kind ------------------------
        for name, spec in cfg.extra_fields.items():
            if spec["column"] not in df.columns:
                raise ValueError(
                    f"extra field {name!r} reads column {spec['column']!r}, which is "
                    f"not in the data. Available: {sorted(df.columns)}"
                )
            df[self._extra_column(name)] = get_field_kind(spec["kind"]).prepare(
                df[spec["column"]], self
            )

        # -- optional identity hashes ---------------------------------
        if cfg.include_customer_hash:
            df[CUSTH_COL] = stable_hash(self._clean_str(df[s.customer_id]))
        if cfg.include_account_hash and cfg.account_token == "order":
            if ACCOUNT_ORDER_COL not in df.columns:
                raise ValueError(
                    f"account_token='order' needs the {ACCOUNT_ORDER_COL!r} column, which "
                    "the data does not have. Add it to the split files once with "
                    "`python scripts/add_account_order.py` (notebook 01 writes it for "
                    "new splits)."
                )
            order = df[ACCOUNT_ORDER_COL]
            if bool(order.isnull().any()):
                raise ValueError(f"{ACCOUNT_ORDER_COL!r} has missing values: rerun "
                                 "`python scripts/add_account_order.py --force`")
            df[ACCT_COL] = order.astype("int32")
        elif cfg.include_account_hash:
            df[ACCTH_COL] = stable_hash(self._clean_str(df[s.account_number]))

        return df

    # -- preprocessing helpers -----------------------------------------

    def _gap_seconds(self, delta):
        """Gap in seconds, clipped at 0; missing for a series' first row."""
        seconds = delta.dt.total_seconds().clip(0)
        if self.config.time_delta_first_token:
            return seconds
        return seconds.fillna(0)

    @staticmethod
    def _to_float(series: cudf.Series) -> cudf.Series:
        """Cast to float, tolerating currency symbols and thousands separators.

        Assumes ``.`` as the decimal separator: ``"EUR 1,234.56"`` -> 1234.56.
        Convert comma-decimal exports before loading (or hand this method a
        float column, which is the fast path).
        """
        if series.dtype == "object":
            cleaned = (
                series.astype("str")
                .str.replace(r"[^0-9.,\-]", "", regex=True)   # drop currency symbols
                .str.replace(",", "", regex=False)            # drop thousands separators
            )
            cleaned = cleaned.where(cleaned != "", "0")
            return cleaned.astype("float64").fillna(0.0)
        return series.astype("float64").fillna(0.0)

    @staticmethod
    def _clean_str(series: cudf.Series) -> cudf.Series:
        """Uppercase, trim, replace whitespace runs, and fill nulls with ``NA``.

        Whitespace must not survive: token strings are joined by spaces in the
        training corpus.
        """
        out = series.astype("str").fillna("NA")
        out = out.str.strip().str.upper().str.replace(r"\s+", "-", regex=True)
        return out.where(out != "", "NA")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def tokens_per_transaction(self) -> int:
        return len(self.field_names)

    def recommended_chunk_size(self, context_window: int = 4096) -> int:
        """Transactions per corpus line for a given context window.

        One separator per transaction plus ``<bos>``/``<eos>``.
        """
        per_txn = self.tokens_per_transaction + 1
        return max(1, (context_window - 2) // per_txn)

    def token_legend(self, max_examples: int = 8) -> Dict[str, Dict[str, Any]]:
        """Per-step vocabulary summary, for sanity-checking a fitted pipeline."""
        legend: Dict[str, Dict[str, Any]] = {}
        for tok_id in self.tokenizer_order:
            step = self.steps[tok_id]
            tokens = list((step._idx_to_token or {}).values())
            legend[tok_id] = {
                "step": type(step).__name__,
                "vocab_size": step.vocab_size,
                "offset": self.vocab_offset.get(tok_id),
                "examples": tokens[:max_examples],
            }
        return legend

    def describe(self) -> str:
        lines = [
            f"{type(self).__name__}: {self.tokens_per_transaction} tokens/transaction, "
            f"vocab_size={self.global_vocab_size}"
            f"{'' if self.is_fitted else ' (NOT FITTED)'}",
            f"  sequence scope: {self.config.sequence_scope} -> key {self.group_cols}",
        ]
        for tok_id, info in self.token_legend(4).items():
            lines.append(
                f"  {tok_id:<16} {info['step']:<26} "
                f"size={info['vocab_size']:<6} offset={info['offset']} "
                f"e.g. {', '.join(map(str, info['examples']))}"
            )
        for tok_id in self.tokenizer_order:
            step = self.steps[tok_id]
            if isinstance(step, HybridHashTokenizer) and self.is_fitted:
                coverage = step.row_coverage
                lines.append(
                    f"  {tok_id}: {step.top_values} counterparties with a dedicated "
                    f"token (>= {step.min_customers} customers each)"
                    + ("" if coverage is None else f", {coverage:.1%} of training rows")
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_metadata(self) -> Dict[str, Any]:
        return {
            "schema": asdict(self.schema),
            "config": asdict(self.config),
            "tokens_per_transaction": self.tokens_per_transaction,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "BankTransferTokenizerPipeline":
        meta = state.get("metadata") or {}
        schema_kwargs = dict(meta.get("schema") or {})
        schema_kwargs.pop("REQUIRED_FIELDS", None)
        schema = BankSchema(**schema_kwargs)
        # Options the saved state predates get the value that reproduces how it
        # was fitted, not today's default.
        config_kwargs = dict(BankTokenizerConfig.LEGACY_DEFAULTS)
        config_kwargs.update(meta.get("config") or {})
        config = BankTokenizerConfig(**config_kwargs)

        pipeline = cls(
            schema=schema,
            config=config,
            special_tokens=dict(state["special_tokens"]),
        )
        pipeline._load_steps(state, strict=True)
        return pipeline


# ── Account number within the customer ────────────────────────────────

def account_order_table(df, schema: BankSchema):
    """One row per (customer, account): ``account_order`` = 1 for the customer's
    first account, 2 for the next one, ... in order of first transaction.

    Pass the customer's FULL history (all splits): computed per split or per
    window, the same account would get different numbers. It only looks back --
    an account's number depends on the accounts that appeared before it, never
    on later activity -- so computing it over all periods leaks nothing. Ties
    (two accounts first used at the same moment) go by account number. Works on
    pandas and cuDF frames; the datetime column must already be a datetime.
    """
    cust, acct, time = schema.customer_id, schema.account_number, schema.datetime
    if df[time].dtype.kind != "M":
        raise ValueError(f"{time!r} must be a datetime column, got {df[time].dtype}")
    first = df[[cust, acct, time]].groupby([cust, acct], as_index=False)[time].min()
    first = first.sort_values([cust, time, acct]).reset_index(drop=True)
    first[ACCOUNT_ORDER_COL] = (first.groupby(cust).cumcount() + 1).astype("int32")
    return first[[cust, acct, ACCOUNT_ORDER_COL]]


def add_account_order(df, schema: BankSchema, table=None):
    """``df`` with the ``account_order`` column, rows kept in their order.

    ``table`` is :func:`account_order_table` over the full history; when None it
    is computed from ``df``, which must then hold that full history.
    """
    if table is None:
        table = account_order_table(df, schema)
    keys = [schema.customer_id, schema.account_number]
    out = df.drop(columns=[ACCOUNT_ORDER_COL]) if ACCOUNT_ORDER_COL in df.columns else df
    out = out.assign(_pos=np.arange(len(out)))
    out = out.merge(table, on=keys, how="left").sort_values("_pos")
    return out.drop(columns=["_pos"]).reset_index(drop=True)
