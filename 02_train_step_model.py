import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from seqpred.step_data import prep_step_data, StepTokenizer, StepDataset
from seqpred.step_model import StepModel

with open("cfg/step_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

input_files = [config["train_data_path"]]


# Set up data
base_data, vocab = prep_step_data(
    data_files=input_files,
    cat_features=(config["ab_cat_feats"], config["pitch_cat_feats"]),
    num_features=(config["ab_num_feats"], config["pitch_num_feats"]),
    n_buckets=config["n_buckets"],
)

tokenizer = StepTokenizer(vocab["features"].to_list())
ds = StepDataset(
    tokenizer=tokenizer,
    df=base_data,
)

train_ds, valid_ds = torch.utils.data.random_split(ds, [0.75, 0.25])

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["n_workers"],
    drop_last=True,
)
valid_dl = torch.utils.data.DataLoader(
    valid_ds,
    batch_size=config["batch_size"],
    num_workers=config["n_workers"],
    drop_last=True,
)

trainer = pl.Trainer(
    max_epochs=config["max_epochs"],
    log_every_n_steps=config["log_every_n"],
    precision=config["precision"],
    logger=pl.loggers.TensorBoardLogger(
        save_dir="./logs", name="step", default_hp_metric=False
    ),
    accumulate_grad_batches=config["accumulate_batches"],
    num_sanity_val_steps=0,
    callbacks=[
        ModelCheckpoint(
            dirpath="./step_model",
            save_top_k=1,
            monitor="validation_loss",
            enable_version_counter=True,
            filename="{version}-{epoch}-{validation_loss:.4f}",
        ),
    ],
)

# Initialize network
with trainer.init_module():

    if config["checkpoint_path"] is None:
        net = StepModel(
            vocab_size=len(tokenizer),
            pad_index=tokenizer.pad_idx,
            **config["model_params"],
            max_length=ds.max_length,
            optim_lr=config["learning_rate"],
        )
        net.compile()
    else:
        raise NotImplementedError("🖕🖕🖕🖕")

torch.set_float32_matmul_precision("medium")
trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)
