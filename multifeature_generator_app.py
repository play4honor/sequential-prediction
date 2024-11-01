from math import ceil

import yaml
import torch
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
import polars as pl
from morphers import Integerizer, Quantiler
import streamlit as st

from seqpred.data import prep_data, BaseDataset
from seqpred.nn import SequentialMargeNet
from seqpred.diag import rollout

checkpoint_path = "./model/version=0-epoch=11-validation_loss=3.6964.ckpt"
data_files = ["./data/2023_data.parquet"]

st.set_page_config(page_title="Generation Tester", layout="wide")

colors = {
    "ball": "green",
    "blocked_ball": "green",
    "bunt_foul_tip": "red",
    "called_strike": "red",
    "foul": "red",
    "foul_bunt": "red",
    "foul_pitchout": "red",
    "foul_tip": "red",
    "hit_by_pitch": "blue",
    "hit_into_play": "blue",
    "missed_bunt": "red",
    "pitchout": "green",
    "swinging_strike": "red",
    "swinging_strike_blocked": "red",
}

markers = {
    "4-Seam Fastball": "o",
    "Changeup": "D",
    "Curveball": "v",
    "Cutter": ">",
    "Eephus": "P",
    "Forkball": "D",
    "Knuckle Curve": "v",
    "Knuckleball": "P",
    "Other": "P",
    "Pitch Out": "P",
    "Screwball": "<",
    "Sinker": "o",
    "Slider": ">",
    "Slow Curve": "v",
    "Slurve": ">",
    "Split-Finger": "D",
    "Sweeper": ">",
    "-": "4",
}


def unmorph(pitches, morphers):
    unmorphed_pitches = {}
    for pk, pv in pitches.items():
        if isinstance(morphers[pk], Integerizer):
            reverse_vocab = {v: k for k, v in morphers[pk].vocab.items()}
            vector = pv.tolist()
            # If there's only one feature
            if not isinstance(vector, list):
                vector = [vector]
            unmorphed_pitches[pk] = [reverse_vocab.get(item, "-") for item in vector]
        else:
            vector = pv.tolist()
            if not isinstance(vector, list):
                vector = [vector]
            qs = morphers[pk].quantiles
            unmorphed_pitches[pk] = [qs[ceil(item * len(qs))] for item in vector]
    return unmorphed_pitches


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
        group_by_cols=["game_pk"],
        rename=config["rename"],
        fixed_cols=fixed_inputs,
        cols=inputs,
        morphers=morpher_dict,
    )
    ds = BaseDataset(
        data,
        config["keys"],
        morpher_dict,
        model.hparams["max_length"],
    )

    return config, model, morpher_dict, ds


config, model, morpher_dict, ds = load_model_and_data(
    "cfg/config.yaml",
    checkpoint_path,
    data_files,
)

with st.sidebar:
    pitch_index = st.number_input("Game Index", 0, len(ds) - 1)
    example = ds[pitch_index]
    after_n_pitches = st.slider("Context_length", 0, 200, value=50, step=1) + 1
    temperature = st.slider("Generation Temperature", 0.0, 2.0, value=1.0, step=0.01)
    if st.button("Re-run"):
        st.rerun()

with torch.inference_mode():
    # noooooooo
    inning_mask = (
        torch.arange(example["description"].shape[0]) < after_n_pitches
    ) & ~torch.isinf(example["pad_mask"])

    x = {
        k: v[inning_mask].to(model.device).unsqueeze(0)
        for k, v in example.items()
        if isinstance(v, torch.Tensor)
    }
    n_generated = None
    attention_per_step = []
    for i in range(512 - after_n_pitches):
        generated_pitch = model.generate_one(
            x, keep_attention=True, temperature=temperature
        )
        x = {
            k: torch.cat([v, generated_pitch[k].unsqueeze(0)], dim=1)
            for k, v in x.items()
            if k != "pad_mask"
        }
        # Get attention activations.
        attention = [
            torch.nn.functional.softmax(layer.gq_attn.attention_activation, dim=-1)
            for layer in model.transformer.transformer_layers
        ]
        attention_per_step.append(rollout(attention, head_fusion="mean"))
        if (
            generated_pitch["end_of_game"].item()
            == morpher_dict["end_of_game"].vocab[True]
        ):
            n_generated = i + 1
            print(f"Reached end of game: generated {i+1} pitches")
            break

    n_generated = 512 if n_generated is None else n_generated

    context = {k: v[:, :after_n_pitches] for k, v in x.items()}
    generated = {k: v[:, after_n_pitches:] for k, v in x.items()}
    # yuck
    attention_at_each_step = torch.zeros(
        [n_generated, n_generated + after_n_pitches]
    ).to(attention_per_step[0])

    for i in range(n_generated - 1):
        attention_at_each_step[i, : i + after_n_pitches] = attention_per_step[i]

context_df = pl.DataFrame(
    {"source": "context"}
    | unmorph(
        # [1:] removes the start token.
        {k: v[:, 1:].squeeze().cpu().numpy() for k, v in context.items()},
        morpher_dict,
    )
)
generated_df = pl.DataFrame(
    {"source": "generated"}
    | unmorph(
        {k: v.squeeze().cpu().numpy() for k, v in generated.items()},
        morpher_dict,
    )
)
if context_df.height > 0:
    pitch_df = pl.concat([context_df, generated_df]).with_row_index(offset=0)
else:
    pitch_df = generated_df.with_row_index(offset=0)

pitch_df = pitch_df.with_columns(
    empirical_inning=0.5 + 0.5 * pl.col("end_of_inning").cum_sum()
)

col1, col2 = st.columns([0.99, 0.01])

with col1:

    st.markdown("#### Pitches")
    st.dataframe(pitch_df, hide_index=True, width=1_000_000)

    fig, ax = plt.subplots(1)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    sns.heatmap(
        attention_at_each_step.cpu().numpy(),
        ax=ax,
        linewidth=0.0,
        cmap=sns.cubehelix_palette(as_cmap=True),
    )
    st.markdown("#### Attention")
    st.pyplot(fig)

with st.sidebar:
    if n_generated is not None:
        st.markdown(f"Generated {n_generated} pitches")
    else:
        st.markdown("Reached generation limit")
