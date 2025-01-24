import torch
from torch import nn
from torchmetrics.text import Perplexity
from lightning import pytorch as pl

from .nn import BoringPositionalEncoding
from .cmlk import Transformer, RMSNorm


class StepModel(pl.LightningModule):

    def __init__(
        self,
        vocab_size: int,
        pad_index: int,
        d_model: int,
        max_length: int,
        n_layers: int,
        ff_dim: int,
        n_heads: int,
        optim_lr: float = 0.001,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        self.d_model = d_model
        self.max_length = max_length
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.n_heads = n_heads
        self.transformer_layer_args = {
            "d_model": self.d_model,
            "ff_dim": self.ff_dim,
        }
        self.attn_args = {
            "n_kv_heads": self.n_heads,
            "n_q_heads": self.n_heads,
        }

        # Optimizer
        self.optim_lr = optim_lr

        # Model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_encoding = BoringPositionalEncoding(self.max_length, self.d_model)
        self.transformer = Transformer(
            self.n_layers,
            layer_args=self.transformer_layer_args,
            position_encoding="nope",
            attn_args=self.attn_args,
        )
        self.prediction_head = nn.Sequential(
            RMSNorm(self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.vocab_size),
        )
        # Initialize linear layers and embeddings, at least.
        self.apply(self._init_weights)

        # loss
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_index)

        # Metrics
        self.train_perplexity = Perplexity(ignore_index=self.pad_index)
        self.validation_perplexity = Perplexity(ignore_index=self.pad_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.transformer(x)
        x = self.prediction_head(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    def training_step(self, x):

        # (n, s, v)
        y_hat = self(x)[:, :-1, :]
        loss = self.loss_func(y_hat.permute(0, 2, 1), x[:, 1:])
        self.log("train_loss", loss)
        self.log("train_perplexity", self.train_perplexity(y_hat, x[:, 1:]))
        return loss

    def validation_step(self, x):

        # (n, s, v)
        y_hat = self(x)[:, :-1, :]
        loss = self.loss_func(y_hat.permute(0, 2, 1), x[:, 1:])
        self.log("validation_loss", loss)
        self.log("validation_perplexity", self.validation_perplexity(y_hat, x[:, 1:]))
        return loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
