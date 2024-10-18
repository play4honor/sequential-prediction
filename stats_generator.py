from math import ceil

import yaml
import torch
import polars as pl
from morphers import Integerizer, Quantiler
import streamlit as st

from seqpred.data import prep_data, BaseDataset
from seqpred.nn import SequentialMargeNet

checkpoint_path = "./model/epoch=14-validation_loss=10.321.ckpt"
data_files = ["./data/2023_data.parquet"]

st.set_page_config(page_title="Synthetic Statistics", layout="wide")


def unmorph_numeric(x, morpher):
    qs = morpher.quantiles
    return (x * len(qs)).ceil().replace({i: v for i, v in enumerate(qs)})


@st.cache_resource
def load_model_and_data(config_path, checkpoint_path, data_path):

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    morpher_dispatch = {
        "numeric": Quantiler,
        "categorical": Integerizer,
    }

    fixed_inputs = {
        col: (morpher_dispatch[tp], kwargs)
        for [col, tp, kwargs] in config["fixed_features"]
    }

    inputs = {
        col: (morpher_dispatch[tp], kwargs) for [col, tp, kwargs] in config["features"]
    }

    model = SequentialMargeNet.load_from_checkpoint(checkpoint_path)

    morpher_dict = model.hparams["morphers"]

    # Set up data
    data, _ = prep_data(
        data_files=data_path,
        group_by=["game_pk", "at_bat_number"],
        rename=config["rename"],
        fixed_cols=fixed_inputs,
        cols=inputs,
        morphers=morpher_dict,
        extra_labels={
            "pitcher_name": "pitcher_reference",
            "batter_name": "batter_reference",
        },
    )

    return config, model, morpher_dict, data


config, model, morpher_dict, data = load_model_and_data(
    "cfg/config.yaml",
    checkpoint_path,
    data_files,
)

# Get pitcher and batter lists
pitchers = sorted(data["pitcher_reference"].unique().to_list())
batters = sorted(data["batter_reference"].unique().to_list())

with st.sidebar:

    prog_bar = st.progress(0, "Done")

    player_type = st.selectbox("Player Type", ["Pitchers", "Batters"])

    if player_type == "Pitchers":
        player = st.selectbox("Pitcher", pitchers)
        filter_column = "pitcher_reference"
    else:
        player = st.selectbox("Batter", batters)
        filter_column = "batter_reference"

    temperature = st.slider("Generation Temperature", 0.0, 2.0, value=1.0, step=0.01)
    n_samples = st.slider("Number of Samples", 1, 10, 3, 1)
    if st.button("Re-run"):
        st.rerun()

pitcher_data = data.filter(pl.col(filter_column) == player)

ds = BaseDataset(
    pitcher_data,
    config["keys"],
    morpher_dict,
    model.hparams["max_length"],
)

all_df = []
for i in range(n_samples):
    prog_bar.progress(int(i * (100 / n_samples)), text="Generating")

    dl = torch.utils.data.DataLoader(ds, batch_size=2048)

    with torch.inference_mode():

        # dl will always have one batch.
        batch = next(iter(dl))
        x = {
            k: v[:, 0].to(model.device).unsqueeze(1)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor) and k not in ["game_pk", "at_bat_number"]
        }
        for i in range(19):
            generated_pitch = model.generate_one(
                x, keep_attention=False, temperature=temperature
            )
            x = {
                k: torch.cat([v, generated_pitch[k].unsqueeze(-1)], dim=1)
                for k, v in x.items()
                if k != "pad_mask"
            }
        # Remove the start of sequence positions.
        x = {k: v[:, 1:] for k, v in x.items()}

        x |= {
            "game_pk": batch["game_pk"],
            "at_bat_number": batch["at_bat_number"],
            "end_position": x["end_of_at_bat"].argmax(dim=1),
        }

    generated_df = pl.DataFrame({k: v.detach().cpu().numpy() for k, v in x.items()})
    truncated_df = generated_df.with_columns(
        *[
            pl.col(k).list.slice(0, pl.col("end_position") + 1)
            for k, v in generated_df.schema.items()
            if v == pl.List
        ]
    )

    exploded_df = truncated_df.explode(
        col
        for col in truncated_df.columns
        if col not in ["game_pk", "at_bat_number", "end_position"]
    )
    all_df.append(exploded_df)

prog_bar.progress(100, text="Finished!")

# Some unmorph stuff
pn_dict = {v: k for k, v in morpher_dict["pitch_name"].vocab.items()}
desc_dict = {v: k for k, v in morpher_dict["description"].vocab.items()}
events_dict = {v: k for k, v in morpher_dict["events"].vocab.items()}

complete_df = pl.concat(all_df)

if player_type == "Pitchers":

    summary_df = (
        complete_df.with_columns(
            pitch_name=pl.col("pitch_name").replace(pn_dict, default=None),
            description=pl.col("description").replace(desc_dict, default=None),
            plate_x=unmorph_numeric(pl.col("plate_x"), morpher_dict["plate_x"]),
            plate_z=unmorph_numeric(pl.col("plate_z"), morpher_dict["plate_z"]),
            release_speed=unmorph_numeric(
                pl.col("release_speed"), morpher_dict["release_speed"]
            ),
        )
        .with_columns(
            in_zone=pl.when(
                pl.col("plate_x").is_between(-0.71, 0.71),
                pl.col("plate_z").is_between(1.5, 3.5),
            )
            .then(1)
            .otherwise(0),
            is_whiff=pl.col("description").is_in(
                ["swinging_strike", "swinging_strike_blocked", "missed_bunt"]
            ),
            is_swing=pl.col("description").is_in(
                [
                    "swinging_strike",
                    "swinging_strike_blocked",
                    "missed_bunt",
                    "foul",
                    "foul_tip",
                    "foul_bunt",
                    "foul_pitchout",
                    "hit_into_play",
                ]
            ),
        )
        .group_by("pitch_name")
        .agg(
            count=pl.len(),
            average_speed=pl.col("release_speed").mean(),
            n_zone=pl.col("in_zone").sum(),
            whiff_percent=pl.col("is_whiff").sum() / pl.col("is_swing").sum(),
        )
        .with_columns(
            zone_pct=pl.col("n_zone") / pl.col("count"),
            percent=pl.col("count") / pl.col("count").sum(),
            count=(pl.col("count") / n_samples).round(),
            n_zone=pl.col("n_zone") / n_samples,
        )
        .drop("n_zone")
        .sort("percent", descending=True)
    )
    st.dataframe(summary_df)

else:

    summary_df = (
        complete_df.with_columns(
            pitch_name=pl.col("pitch_name").replace(pn_dict, default=None),
            description=pl.col("description").replace(desc_dict, default=None),
            plate_x=unmorph_numeric(pl.col("plate_x"), morpher_dict["plate_x"]),
            plate_z=unmorph_numeric(pl.col("plate_z"), morpher_dict["plate_z"]),
            release_speed=unmorph_numeric(
                pl.col("release_speed"), morpher_dict["release_speed"]
            ),
            events=pl.col("events").replace(events_dict, default=None),
        )
        .filter(
            pl.col("events").is_not_null(),
            ~pl.col("events").str.contains("caught_stealing"),
        )
        .group_by("events")
        .len("N")
        .sort("N", descending=True)
    )
    total_events = summary_df["N"].sum()
    summary_dict = summary_df.to_dict(as_series=False)
    rd = {event: n for event, n in zip(summary_dict["events"], summary_dict["N"])}

    st.metric("K%", f"{round(rd['strikeout'] * 100 / total_events, 1)}%")
    st.metric(
        "Walk%",
        f"{round((rd.get('walk', 0) + rd.get('hit_by_pitch', 0)) * 100 / total_events, 1)}%",
    )
    st.metric("HR", round(rd.get("home_run", 0) / n_samples))
    ba = sum(rd.get(k, 0) for k in ["single", "double", "triple", "home_run"]) / sum(
        rd.get(k, 0)
        for k in [
            "single",
            "double",
            "triple",
            "home_run",
            "field_out",
            "force_out",
            "fielders_choice",
            "fielders_choice_out",
            "grounded_into_double_play",
            "other_out",
            "strikeout",
            "strikeout_double_play",
            "triple_play",
        ]
    )
    st.metric("BA", round(ba, 3))
