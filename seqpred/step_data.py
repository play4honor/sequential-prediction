from itertools import chain

import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset


def make_bucket_limits(x: pl.Series, n_buckets):

    q = np.linspace(0.01, 0.99, n_buckets)
    breaks = np.nanquantile(x.to_numpy(), q)
    # Make sure every value is distinct.
    breaks += np.linspace(0, 1e-5, n_buckets)
    breaks_list = breaks.tolist()
    labels = (
        [f"<{breaks[0] :.2f}"]
        + [
            f"{left :.2f}-{right :.2f}"
            for left, right in zip(breaks_list[:-1], breaks_list[1:])
        ]
        + [f">{breaks_list[-1] :.2f}"]
    )
    return breaks, labels


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

    starting_pitchers = (
        input_data.sort(["at_bat_number"], descending=False)
        .with_columns(
            team=pl.when(pl.col("inning_topbot") == "Bot")
            .then(pl.lit("<AWAY>"))
            .otherwise(pl.lit("<HOME>")),
        )
        .group_by(["game_pk", "team"], maintain_order=True)
        .agg(
            pl.concat_str(
                pl.lit("pitcher_name: "), pl.col("pitcher_name").first()
            ).alias("pitcher_name")
        )
    )

    batting_orders = (
        input_data.sort(["at_bat_number"], descending=False)
        .unique(["game_pk", "inning_topbot", "batter_name"], keep="first")
        .with_columns(
            batting_order=pl.cum_count("at_bat_number").over(
                partition_by=["game_pk", "inning_topbot"], order_by="at_bat_number"
            ),
            team=pl.when(pl.col("inning_topbot") == "Bot")
            .then(pl.lit("<HOME>"))
            .otherwise(pl.lit("<AWAY>")),
        )
        .filter(pl.col("batting_order") <= 9)
        .sort(["batting_order"])
        .group_by(["game_pk", "team"], maintain_order=True)
        .agg(
            pl.concat_str(pl.lit("batter_name: "), pl.col("batter_name")).alias(
                "batter_name"
            )
        )
        .join(starting_pitchers, on=["game_pk", "team"])
        .with_columns(
            lineup=pl.concat_list(
                pl.col("team"), pl.col("pitcher_name"), pl.col("batter_name")
            )
        )
        .sort(["team"])
        .group_by(["game_pk"], maintain_order=True)
        .agg(pl.col("lineup").flatten())
    )

    input_data = (
        input_data
        # Quantize numeric features
        .with_columns(
            **{
                feature: pl.col(feature)
                .cut(
                    quantiles[feature][0],
                    labels=quantiles[feature][1],
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
        .join(batting_orders, on="game_pk")
        .select(
            "game_pk",
            features=pl.concat_list(pl.col("lineup"), pl.col("features")),
            length=pl.col("length") + pl.col("lineup").list.len(),
        )
    )

    vocab = input_data.select(
        pl.col("features").list.explode().value_counts(sort=True).struct.unnest()
    )
    return input_data, vocab


class StepTokenizer:

    def __init__(self, vocab, extra_tokens: list | None = None):

        self.vocab = {token: i for i, token in enumerate(vocab)}
        self.extra_tokens = set(
            ["<PAD>", "<UNK>"] + (extra_tokens if extra_tokens is not None else [])
        )
        for token in self.extra_tokens:
            self.vocab[token] = len(self.vocab)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    @property
    def pad_idx(self):
        return self.vocab["<PAD>"]

    def transform(self, x: list):
        return torch.tensor(
            [self.vocab.get(token, self.vocab["<UNK>"]) for token in x],
            dtype=torch.int64,
        )

    def invert(self, x: list):
        return [self.inverse_vocab.get(idx, "<UNK>") for idx in x]


class StepDataset(Dataset):

    def __init__(self, tokenizer: StepTokenizer, df: pl.DataFrame):

        self.tokenizer = tokenizer
        self.df = df
        self.max_length = self.df["features"].list.len().max()

    def __len__(self):
        return self.df.height

    def __getitem__(self, x):
        row = self.df.row(x, named=True)
        vectorized = torch.cat(
            [
                self.tokenizer.transform(row["features"]),
                torch.tensor(
                    [self.tokenizer.pad_idx] * (self.max_length - len(row["features"])),
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )

        return vectorized.type(torch.LongTensor)


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

    tokenizer = StepTokenizer(vocab["features"].to_list())

    ds = StepDataset(tokenizer, data)
    print(ds.max_length)
    print(ds[500])
