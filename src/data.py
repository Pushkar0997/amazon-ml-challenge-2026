from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

REQUIRED_GT_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]

READ_CHUNK_SIZE = 100_000


def read_tsv(path: str | Path) -> pd.DataFrame:

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=3,
    )


def read_tsv_sampled(
    path: str | Path,
    max_rows: int,
    required_ids: set[str] | None = None,
    chunksize: int = READ_CHUNK_SIZE,
    seed: int = 42,
) -> pd.DataFrame:

    path = Path(path)

    required_ids = required_ids or set()

    if max_rows is None:
        return read_tsv(path)

    required_parts = []
    random_parts = []

    random_budget = max_rows

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=3,
        chunksize=chunksize,
    ):

        chunk = chunk[
            REQUIRED_SOURCE_COLUMNS
        ].copy()

        if required_ids:

            required = chunk[
                chunk["entity_id"].isin(required_ids)
            ]

            if not required.empty:
                required_parts.append(required)

        if random_budget > 0:

            take = min(
                random_budget,
                max(
                    1,
                    min(
                        len(chunk),
                        max_rows // 5 or 1,
                    ),
                ),
            )

            if take > 0:

                sampled = chunk.sample(
                    n=take,
                    random_state=seed,
                )

                random_parts.append(sampled)

                random_budget -= len(sampled)

        if random_budget <= 0 and not required_ids:
            break

    if required_parts:

        required_df = (
            pd.concat(
                required_parts,
                ignore_index=True,
            )
            .drop_duplicates("entity_id")
        )

    else:

        required_df = pd.DataFrame(
            columns=REQUIRED_SOURCE_COLUMNS
        )

    if random_parts:

        random_df = (
            pd.concat(
                random_parts,
                ignore_index=True,
            )
            .drop_duplicates("entity_id")
        )

    else:

        random_df = pd.DataFrame(
            columns=REQUIRED_SOURCE_COLUMNS
        )

    required_set = set(
        required_df["entity_id"]
    )

    random_df = random_df[
        ~random_df["entity_id"].isin(
            required_set
        )
    ]

    remaining = max(
        0,
        max_rows - len(required_df),
    )

    if len(random_df) > remaining:

        random_df = random_df.head(
            remaining
        )

    output = pd.concat(
        [
            required_df,
            random_df,
        ],
        ignore_index=True,
    )

    return (
        output
        .drop_duplicates("entity_id")
        .reset_index(drop=True)
    )


def validate_source(
    df: pd.DataFrame,
    path: str,
):

    missing = [
        c
        for c in REQUIRED_SOURCE_COLUMNS
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    return df[
        REQUIRED_SOURCE_COLUMNS
    ].copy()


def validate_ground_truth(
    df: pd.DataFrame,
    path: str,
):

    missing = [
        c
        for c in REQUIRED_GT_COLUMNS
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    return df[
        REQUIRED_GT_COLUMNS
    ].copy()


def _matched_ids_for_source1(
    gt: pd.DataFrame,
    source1_ids: set[str],
):

    target_ids = set()

    subset = gt[
        gt["source1_entity_id"].isin(
            source1_ids
        )
    ]

    for value in subset[
        "matched_entity_ids"
    ]:

        for entity_id in str(
            value
        ).split(","):

            entity_id = entity_id.strip()

            if entity_id:
                target_ids.add(entity_id)

    return target_ids


def load_split(
    data_dir: str | Path,
    split: str,
    max_s1: int | None = None,
    max_target_per_source: int | None = None,
):

    data_dir = Path(data_dir) / split

    prefix = (
        "train"
        if split == "train"
        else "test"
    )

    if max_s1 is None:

        s1 = validate_source(
            read_tsv(
                data_dir
                / f"{prefix}_source1.tsv"
            ),
            f"{prefix}_source1.tsv",
        )

    else:

        s1 = validate_source(
            read_tsv_sampled(
                data_dir
                / f"{prefix}_source1.tsv",
                max_s1,
            ),
            f"{prefix}_source1.tsv",
        )

        s1 = (
            s1
            .head(max_s1)
            .reset_index(drop=True)
        )

    if split == "train":

        gt = validate_ground_truth(
            read_tsv(
                data_dir
                / "train_ground_truth.tsv"
            ),
            "train_ground_truth.tsv",
        )

    else:

        gt = None

    source1_ids = set(
        s1["entity_id"]
    )

    if gt is not None:

        required_target_ids = (
            _matched_ids_for_source1(
                gt,
                source1_ids,
            )
        )

    else:

        required_target_ids = set()

    if max_target_per_source is None:

        s2 = validate_source(
            read_tsv(
                data_dir
                / f"{prefix}_source2.tsv"
            ),
            f"{prefix}_source2.tsv",
        )

        s3 = validate_source(
            read_tsv(
                data_dir
                / f"{prefix}_source3.tsv"
            ),
            f"{prefix}_source3.tsv",
        )

    else:

        s2 = validate_source(
            read_tsv_sampled(
                data_dir
                / f"{prefix}_source2.tsv",
                max_target_per_source,
                required_ids=required_target_ids,
            ),
            f"{prefix}_source2.tsv",
        )

        s3 = validate_source(
            read_tsv_sampled(
                data_dir
                / f"{prefix}_source3.tsv",
                max_target_per_source,
                required_ids=required_target_ids,
            ),
            f"{prefix}_source3.tsv",
        )

    if split == "train":

        return (
            s1,
            s2,
            s3,
            gt,
        )

    return (
        s1,
        s2,
        s3,
    )


def add_normalized_columns(
    df: pd.DataFrame,
):
    """Attach src.preprocess.normalize()'s columns (name_core, name_sorted,
    name_compact, name_translit, legal_form, is_domain, addr_norm,
    addr_components, addr_numbers, state_canon, city_canon, script) to df.
    Replaces the old src.utils ASCII-only normalizer, which silently dropped
    non-Latin names entirely (the root cause of the India name-Jaccard=0
    problem — see AGENT_LOG.md)."""
    from .preprocess import normalize

    records = [
        normalize(name, address, country)
        for name, address, country in zip(
            df["business_name"], df["business_address"], df["country"]
        )
    ]
    norm_df = pd.DataFrame.from_records(records, index=df.index).drop(columns=["country"])

    output = pd.concat([df, norm_df], axis=1)
    output["country_norm"] = output["country"].map(lambda x: str(x).strip().lower())

    return output


def sample_dataframe(
    df: pd.DataFrame,
    max_rows: int,
    required_ids: set[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Cap an already-loaded DataFrame to max_rows, keeping every row whose
    entity_id is in required_ids (e.g. a ground-truth match target) even if
    that pushes the count slightly over max_rows, then filling the rest with
    a random sample. Used for --preprocessed-dir dev runs, where the parquet
    is already fully loaded and load_split's streaming/sampling doesn't apply."""
    if len(df) <= max_rows:
        return df.reset_index(drop=True)

    required_ids = required_ids or set()
    required_df = df[df["entity_id"].isin(required_ids)]
    remaining = df[~df["entity_id"].isin(required_ids)]

    take = max(0, max_rows - len(required_df))
    random_df = remaining.sample(n=min(take, len(remaining)), random_state=seed) if take else remaining.iloc[0:0]

    return (
        pd.concat([required_df, random_df], ignore_index=True)
        .drop_duplicates("entity_id")
        .reset_index(drop=True)
    )