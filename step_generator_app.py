from math import ceil, isnan

import yaml
import torch
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
import polars as pl
import streamlit as st

from seqpred.step_data import StepPrep, StepTokenizer, StepDataset
from seqpred.step_model import StepModel
from seqpred.diag import rollout

checkpoint_path = "./step_model/version=0-epoch=15-validation_loss=2.2496.ckpt"
data_files = ["./data/2023_data.parquet"]

st.set_page_config(page_title="Generation Tester", layout="wide")

# TKTK THIS IS AWFUL


@st.cache_resource
def load_model_and_data(config_path, checkpoint_path, data_path):

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    input_files = [config["train_data_path"]]

    # Set up data
    step_prep = StepPrep(
        cat_features=(config["ab_cat_feats"], config["pitch_cat_feats"]),
        num_features=(config["ab_num_feats"], config["pitch_num_feats"]),
        n_buckets=config["n_buckets"],
    )

    base_data = step_prep(input_files, overwrite=False)
    vocab = step_prep.vocab

    tokenizer = StepTokenizer(vocab)

    ds = StepDataset(
        tokenizer,
        base_data,
    )

    # Set the max sequence length
    config["model_params"]["pe_args"]["max_seq_len"] = ds.max_length

    # bruh
    model = StepModel.load_from_checkpoint(
        checkpoint_path,
        vocab_size=len(tokenizer),
        pad_index=tokenizer.pad_idx,
        **config["model_params"],
        max_length=ds.max_length,
        optim_lr=config["learning_rate"],
    )

    return config, model, tokenizer, ds


config, model, tokenizer, ds = load_model_and_data(
    "cfg/step_config.yaml",
    checkpoint_path,
    data_files,
)

with st.sidebar:
    eos_token_index = tokenizer.vocab["<EOS>"]
    game_index = st.number_input("Game Index", 0, len(ds) - 1)
    example = ds[game_index].unsqueeze(0)
    after_n_pitches = st.slider("Context_length", 0, 200, value=30, step=1)
    temperature = st.slider("Generation Temperature", 0.0, 2.0, value=1.0, step=0.01)
    if st.button("Re-run"):
        st.rerun()

progress_bar = st.progress(0.0, text="Playing Ball")

with torch.inference_mode():

    # This is padded
    x = example.to(model.device)
    n_generated = None
    attention_per_step = []
    for i in range(ds.max_length - after_n_pitches):
        token_distribution = model(x, keep_attention=True)[0, after_n_pitches + i]
        token_p = torch.nn.functional.softmax(token_distribution / temperature, dim=0)
        selected_token = torch.searchsorted(
            token_p.cumsum(dim=0), torch.rand([1]).to(model.device)
        ).unsqueeze(0)

        x[:, after_n_pitches + i + 1] = selected_token
        # # Get attention activations.
        # attention = [
        #     torch.nn.functional.softmax(layer.gq_attn.attention_activation, dim=-1)
        #     for layer in model.transformer.transformer_layers
        # ]
        # attention_per_step.append(rollout(attention, head_fusion="mean"))
        if selected_token.item() == eos_token_index:
            n_generated = after_n_pitches + i + 1
            print(f"Reached end of game: generated {i+1} tokens")
            break

        progress_bar.progress(
            i / (ds.max_length - after_n_pitches), text="Playing Ball"
        )

    progress_bar.empty()

    n_generated = (
        (ds.max_length - after_n_pitches) if n_generated is None else n_generated
    )

    context = x[:, :after_n_pitches]
    generated = x[:, after_n_pitches:]
    # yuck
    # attention_at_each_step = torch.zeros(
    #     [n_generated, n_generated + after_n_pitches]
    # ).to(attention_per_step[0])

    # for i in range(n_generated - 1):
    #     attention_at_each_step[i, : i + after_n_pitches] = attention_per_step[i]

is_generated = (["context"] * after_n_pitches) + (
    ["generated"] * (n_generated - after_n_pitches)
)

generated_df = pl.DataFrame(
    {
        "is_generated": is_generated,
        "token": tokenizer.invert(x.squeeze(0)[:n_generated].tolist()),
    }
)

pitch_df = generated_df.with_row_index(offset=0)

st.markdown("#### Pitches")
st.dataframe(pitch_df, hide_index=True, width=1_000_000)

# fig, ax = plt.subplots(1)
# fig.set_figwidth(12)
# fig.set_figheight(12)
# sns.heatmap(
#     attention_at_each_step.cpu().numpy(),
#     ax=ax,
#     linewidth=0.0,
#     cmap=sns.cubehelix_palette(as_cmap=True),
# )
# st.markdown("#### Attention")
# st.pyplot(fig)

with st.sidebar:
    if n_generated is not None:
        st.markdown(f"Generated {n_generated} pitches")
    else:
        st.markdown("Reached generation limit")
