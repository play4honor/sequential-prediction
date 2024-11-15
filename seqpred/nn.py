import math

import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl

from .cmlk import Transformer


def log_dict_step(logger, metric_dict, global_step):

    for k, v in metric_dict.items():
        logger.experiment.add_scalar(k, v, global_step=global_step)


class BoringPositionalEncoding(nn.Module):
    """
    Shamelessly "adapted" from a torch tutorial
    """

    def __init__(self, max_length: int, d_model: int):
        super().__init__()

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x + self.pe[:, : x.shape[1], :]


class SumMarginalHead(nn.Module):

    def __init__(
        self,
        morphers: dict,
        hidden_size: int,
    ):
        """Should only expect morphers for the things to predict"""

        super().__init__()
        self.morphers = morphers
        self.embedders = nn.ModuleDict(
            {
                col: morpher.make_embedding(hidden_size)
                for col, morpher in morphers.items()
            }
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.predictors = nn.ModuleDict(
            {
                col: morpher.make_predictor_head(hidden_size)
                for col, morpher in morphers.items()
            }
        )

    def forward(self, input_embedding, features):
        # n x s x k x e
        embeddings = torch.stack(
            [input_embedding]
            + [
                # We don't make a prediction for a context-free first pitch.
                embedder(features[col])[:, 1:, :]
                for col, embedder in self.embedders.items()
            ],
            dim=-2,
        )

        # n x s-1 x k-1 x e
        masked_sums = embeddings.cumsum(dim=-2)[:, :, :-1, :]
        masked_sums = self.activation(self.norm(masked_sums))

        predictions = {
            col: predictor(masked_sums[:, :, i, :])
            for i, (col, predictor) in enumerate(self.predictors.items())
        }

        return predictions

    def embed_inputs(self, features):
        # We actually need to use the input embeddings twice in the process,
        # once at the input to the transformer and once during generation.
        # n x s x k x e
        # (Probably just getting summed, but we'll see.)
        embeddings = torch.stack(
            [embedder(features[col]) for col, embedder in self.embedders.items()],
            dim=-2,
        )
        return embeddings

    def generate(self, x, **kwargs):
        # x is context, n x e
        # embeddings is n x len(predictors) x e
        embeddings = torch.zeros(
            [x.shape[0], len(self.predictors) + 1, x.shape[-1]]
        ).to(x)
        preds = {}
        embeddings[:, 0, :] = x
        for i, (feat, predictor) in enumerate(self.predictors.items()):

            # Predict on the previous context
            total_context = torch.sum(embeddings[:, : i + 1, :], dim=-2)
            total_context = self.activation(self.norm(total_context))

            # Make a draw
            p_dist = predictor(total_context)
            generated_values = self.morphers[feat].generate(p_dist, **kwargs)

            preds[feat] = generated_values
            new_embedding = self.embedders[feat](generated_values)
            embeddings[:, i + 1, :] = new_embedding

        return preds


class SequentialMargeNet(pl.LightningModule):

    def __init__(
        self,
        morphers: dict,
        hidden_size: int,
        max_length,
        optim_lr: float,
        tr_args: dict,
        loss_weights: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optim_lr = optim_lr
        self.max_length = max_length
        self.loss_weights = loss_weights

        self.global_log_step = 0

        # This also includes input embedding layers.
        self.generator_head = SumMarginalHead(
            morphers={col: morpher for col, morpher in morphers.items()},
            hidden_size=hidden_size,
        )

        self.input_norm = nn.GELU()
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(max_length - 1),
        )

        self.transformer = Transformer(
            n_layers=tr_args["n_layers"],
            layer_args=tr_args["layer_args"],
            position_encoding=tr_args["position_encoding"],
            attn_args=tr_args["attn_args"],
        )

        # Initialize linear layers and embeddings, at least.
        self.apply(self._init_weights)

        self.criteria = {
            col: morpher.make_criterion() for col, morpher in morphers.items()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def on_train_start(self):
        if self.loss_weights is not None:
            print("Training using loss weights:")
            for k, v in self.loss_weights.items():
                print(f"{k}: {v:.3f}")

    def forward(self, x):
        # Sequences in x run from the first pitch to the last pitch.
        # Predictions run from the _second_ pitch to the last pitch.
        # n x s-1 x k x e
        tr_inputs = self.generator_head.embed_inputs(x)[:, :-1, :, :]
        # n x s-1 x e
        tr_inputs = tr_inputs.sum(dim=-2)
        tr_inputs = self.input_norm(tr_inputs)
        tr_outputs = self.transformer(tr_inputs, mask=self.causal_mask)
        predictions = self.generator_head(tr_outputs, x)
        return predictions

    def training_step(self, x):
        preds = self(x)
        loss_mask = (~torch.isinf(x["pad_mask"])).float()
        # TKTK use a prefix length config
        loss_mask[:19, :] = 0
        loss_dict = {
            f"train_{col}_loss": (
                criterion(preds[col], x[col][:, 1:]) * loss_mask[:, 1:]
            ).sum()
            / loss_mask.sum()
            for col, criterion in self.criteria.items()
        }
        self.global_log_step += loss_mask.shape[0]
        log_dict_step(self.logger, loss_dict, self.global_log_step)

        if self.loss_weights is not None:
            losses = [
                loss_dict[f"train_{col}_loss"] * self.loss_weights[col]
                for col in self.loss_weights
            ]
        else:
            losses = loss_dict.values()
        total_loss = sum(losses)
        log_dict_step(self.logger, {"train_loss": total_loss}, self.global_log_step)
        return total_loss

    def validation_step(self, x):
        preds = self(x)
        loss_mask = (~torch.isinf(x["pad_mask"])).float()
        loss_dict = {
            f"validation_{col}_loss": (
                criterion(preds[col], x[col][:, 1:]) * loss_mask[:, 1:]
            ).sum()
            / loss_mask.sum()
            for col, criterion in self.criteria.items()
        }
        log_dict_step(self.logger, loss_dict, self.global_log_step)
        total_loss = sum(loss_dict.values())
        log_dict_step(
            self.logger, {"validation_loss": total_loss}, self.global_log_step
        )
        self.log("validation_loss", total_loss, logger=False)
        return total_loss

    def generate_one(self, x, keep_attention: bool = False, **kwargs):
        """Generate a pitch.

        - x is an initial set of pitches
        - kwargs are keyword arguments for the morphers' generate methods"""

        tr_inputs = self.generator_head.embed_inputs(x)

        # n x s-1 x e
        tr_inputs = tr_inputs.sum(dim=-2)
        tr_inputs = self.input_norm(tr_inputs)
        mask = self.causal_mask[: tr_inputs.shape[1], : tr_inputs.shape[1]]
        tr_outputs = self.transformer(
            tr_inputs, mask=mask, keep_attention=keep_attention
        )

        return self.generator_head.generate(tr_outputs[:, -1, :], **kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
