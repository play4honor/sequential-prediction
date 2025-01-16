from itertools import chain

import polars as pl
import numpy as np


def make_bucket_limits(x: pl.Series, n_buckets):

    q = np.linspace(0.01, 0.99, n_buckets)
    breaks = np.nanquantile(x.to_numpy(), q)
    # Make sure every value is distinct.
    breaks += np.linspace(0, 1e-5, n_buckets)
    return breaks


def prep_step_data(
    data_files: list[str],
    cat_features: tuple[list[str], list[str]],
    num_features: tuple[list[str], list[str]],
    n_buckets: int,
    precomputed_quantiles: dict[str, list] | None = None,
):

    input_dataframes = [pl.read_parquet(file) for file in data_files]
    input_data = pl.concat(input_dataframes)

    # Quantile calculation
    quantiles = (
        {
            feature: make_bucket_limits(input_data[feature], n_buckets)
            for feature in num_features[0] + num_features[1]
        }
        if precomputed_quantiles is None
        else precomputed_quantiles
    )

    input_data = (
        input_data
        # Quantize numeric features
        .with_columns(
            **{
                feature: pl.col(feature)
                .cut(
                    quantiles[feature],
                    labels=np.arange(len(quantiles[feature]) + 1).astype("str"),
                )
                .cast(pl.String)
                for feature in quantiles
            }
        )
        .with_columns(
            **{
                feature: pl.concat_str(
                    pl.lit(f"{feature}: "),
                    pl.coalesce(pl.col(feature), pl.lit("MISSING")),
                )
                for feature in chain.from_iterable(cat_features + num_features)
            },
        )
        .select(
            "game_pk",
            "at_bat_number",
            "pitch_number",
            pl.concat_list(
                pl.lit("<AB>"),
                *[pl.col(feat) for feat in num_features[0] + cat_features[0]],
            ).alias("ab_feats"),
            pl.concat_list(
                pl.lit("<PITCH>"), *(cat_features[1] + num_features[1])
            ).alias("pitch_feats"),
        )
        .sort("pitch_number")
        .group_by("game_pk", "at_bat_number", maintain_order=True)
        .agg(
            ab_feats=pl.col("ab_feats").first(),
            pitch_feats=pl.col("pitch_feats").flatten(),
        )
        .with_columns(features=pl.concat_list("ab_feats", "pitch_feats"))
        .sort("at_bat_number")
        .group_by("game_pk", maintain_order=True)
        .agg(features=pl.col("features").flatten())
        .with_columns(
            features=pl.concat_list(
                pl.lit("<SOS>"), pl.col("features"), pl.lit("<EOS>")
            ),
            length=pl.col("features").list.len(),
        )
    )

    vocab = input_data.select(
        pl.col("features").list.explode().value_counts(sort=True).struct.unnest()
    )
    return input_data, vocab


if __name__ == "__main__":

    data, vocab = prep_step_data(
        data_files=["./data/all_data.parquet"],
        cat_features=(
            ["inning", "inning_topbot", "pitcher", "batter"],
            ["pitch_type"],
        ),
        num_features=([], ["release_speed"]),
        n_buckets=32,
    )

    print(vocab)
