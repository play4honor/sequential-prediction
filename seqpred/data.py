import polars as pl
import torch
import yaml


def load_morphers(
    state_path: str,
    cols: dict,
    type_map: dict,
):
    with open(state_path, "r") as f:
        morpher_states = yaml.load(f, Loader=yaml.CLoader)

    morphers = {
        col: type_map[ctype].from_state_dict(morpher_states[col])
        for col, ctype in cols.items()
    }
    return morphers


def pad_tensor_dict(tensor_dict, max_length, return_mask: bool = True):
    """
    Pad a tensor dict up to the max length.
    Padded Location = 0
    """

    init_length = next(iter(tensor_dict.values())).shape[0]
    if init_length >= max_length:
        padded_tensor_dict = {k: v[:max_length] for k, v in tensor_dict.items()}
    else:
        padded_tensor_dict = {
            k: torch.nn.functional.pad(v, [0, max_length - init_length], value=0)
            for k, v in tensor_dict.items()
        }
    # FALSE IS NOT PAD, TRUE IS PAD
    if return_mask:
        pad_mask = torch.ones([max_length], dtype=torch.bool)
        pad_mask[: min(init_length, max_length)] = False
        pad_mask = torch.where(pad_mask, float("-inf"), 0.0)
        return padded_tensor_dict, pad_mask
    else:
        return padded_tensor_dict


def prep_data(
    data_files: str,
    group_by_cols: list,
    rename: dict | None = None,
    fixed_cols: dict | None = None,
    cols: dict | None = None,
    morphers: dict | None = None,
    data_output: str | None = None,
    morpher_output: str | None = None,
    extra_labels: dict = {},
    write: bool = False,
):
    """Prepare data according to a morpher dict.
    The end_of_* features are special and are constructed here."""

    input_dataframes = [pl.read_parquet(file) for file in data_files]
    input_data = pl.concat(input_dataframes)

    if rename is not None:
        input_data = input_data.rename(rename)

    # Create end_of_* indicators
    input_data = (
        input_data.with_columns(
            pl.concat_str(
                pl.col("inning"), pl.col("inning_topbot"), separator="-"
            ).alias("complete_inning"),
            *[pl.col(k).alias(v) for k, v in extra_labels.items()],
        )
        .sort(["game_pk", "at_bat_number", "pitch_number"])
        .with_columns(
            (
                pl.col("complete_inning").over("game_pk")
                != pl.col("complete_inning").over("game_pk").shift(-1, fill_value=-1)
            ).alias("end_of_inning"),
            (
                pl.col("at_bat_number").over("game_pk")
                != pl.col("at_bat_number").over("game_pk").shift(-1, fill_value=-1)
            ).alias("end_of_at_bat"),
            (pl.col("game_pk") != pl.col("game_pk").shift(-1, fill_value=-1)).alias(
                "end_of_game"
            ),
        )
    )

    start_sequence_tokens = input_data.unique(group_by_cols).with_columns(
        pitch_number=pl.lit(-1).cast(pl.Int64),
        **{
            column: (
                pl.lit(None).cast(input_data[column].dtype)
                if (
                    input_data[column].dtype.is_numeric()
                    or input_data[column].dtype.is_(pl.Boolean)
                )
                else pl.lit("start_of_sequence")
            )
            for column in cols.keys()
        },
    )
    input_data = pl.concat([input_data, start_sequence_tokens])

    all_cols = fixed_cols | cols

    # Use existing morpher states
    if morphers is None:
        morphers = {
            feature: morpher_class.from_data(input_data[feature], **kwargs)
            for feature, (morpher_class, kwargs) in all_cols.items()
        }
    else:
        morphers = morphers

    input_data = (
        input_data.select(
            "game_pk",
            "at_bat_number",
            "pitch_number",
            *list(extra_labels.values()),
            # morphed inputs
            *[
                morpher(morpher.fill_missing(pl.col(feature))).alias(feature)
                for feature, morpher in morphers.items()
            ],
        )
        .sort(["at_bat_number", "pitch_number"])
        .group_by(group_by_cols + list(extra_labels.values()), maintain_order=True)
        .agg(
            *[pl.col(feature) for feature in morphers.keys()],
            n_pitches=pl.col("pitch_number").count(),
        )
    )

    if write:
        input_data.write_parquet(data_output)
        morpher_dict = {
            column: morpher.save_state_dict() for column, morpher in morphers.items()
        }
        with open(morpher_output, "w") as f:
            yaml.dump(morpher_dict, f)

    return input_data, morphers


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        ds: pl.DataFrame,
        keys: list,
        morphers: dict,
        max_length: int,
    ):

        super().__init__()
        self.ds = ds
        self.keys = keys
        self.morphers = morphers
        self.max_length = max_length

    def __len__(self):
        return self.ds.height

    def __getitem__(self, idx):
        row = self.ds.row(idx, named=True)
        inputs, pad_mask = pad_tensor_dict(
            {
                k: torch.tensor(row[k], dtype=morpher.required_dtype)
                for k, morpher in self.morphers.items()
            },
            max_length=self.max_length,
        )

        inputs |= {key: row[key] for key in self.keys}
        inputs |= {"pad_mask": pad_mask}
        return inputs
