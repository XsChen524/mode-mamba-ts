import torch
import torch.nn as nn
from src.mamba_mode.modules.mamba_mode import Mamba
from src.mamba_mode.modules.mamba_replacement import (
    SMambaBlock,
    AttentionFFNBlock,
    LinearBlock,
)

from src.layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Encoder: stack of layers based on replace_block parameter
        replace_block = getattr(configs, "replace_block", "none")
        # Set hippo_init based on --hippo parameter (default: False)
        hippo_init = getattr(configs, "hippo", False)

        # Embedding (inverted for multivariate time series)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        if replace_block == "none":
            # Use ODE-enhanced Mamba (original behavior)
            # Determine low-rank ODE parameters based on ode_type
            ode_type = getattr(configs, "ode_type", "dynamic")

            # Switch logic to set parameters based on ode_type
            if ode_type == "dynamic":
                low_rank_ode = True
                static_low_rank = False
            elif ode_type == "static":
                low_rank_ode = True
                static_low_rank = True
            elif ode_type == "none":
                low_rank_ode = False
                static_low_rank = False
            else:
                raise ValueError(
                    f"Unknown ode_type: {ode_type}. Must be one of ['static', 'dynamic', 'none']"
                )

            self.encoder = nn.ModuleList(
                [
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=configs.d_conv,
                        expand=configs.expand,
                        r_rank=configs.r_rank,
                        low_rank_ode=low_rank_ode,
                        static_low_rank=static_low_rank,
                        hippo_init=hippo_init,
                    )
                    for _ in range(configs.e_layers)
                ]
            )
        elif replace_block == "s-mamba":
            # Use s-mamba block (bi-mamba + FFN TD Encoding)
            self.encoder = nn.ModuleList(
                [
                    SMambaBlock(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=configs.d_conv,
                        expand=configs.expand,
                        dropout=configs.dropout,
                    )
                    for _ in range(configs.e_layers)
                ]
            )
        elif replace_block == "attention-ffn":
            # Use Attention + FFN block
            self.encoder = nn.ModuleList(
                [
                    AttentionFFNBlock(
                        d_model=configs.d_model,
                        n_heads=8,  # Default number of heads
                        d_ff=configs.d_model * 4,  # Typical FFN expansion
                        dropout=configs.dropout,
                    )
                    for _ in range(configs.e_layers)
                ]
            )
        elif replace_block == "linear":
            # Use simple Linear block
            self.encoder = nn.ModuleList(
                [
                    LinearBlock(
                        d_model=configs.d_model,
                        dropout=configs.dropout,
                    )
                    for _ in range(configs.e_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown replace_block value: {replace_block}")

        self.norm = torch.nn.LayerNorm(configs.d_model)

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting method implementing the MODE algorithm:

        Returns the predicted future time series Ŷ = (y_{L+1}, ..., y_{L+H})
        as specified in Step 15 of the pseudocode.
        """
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder forward pass
        for layer in self.encoder:
            enc_out = layer(enc_out)

        enc_out = self.norm(enc_out)

        proj = self.projector(enc_out)  # (B, N, pred_len)
        future_pred = proj.permute(0, 2, 1)[
            :, :, : self.configs.c_out
        ]  # keep first c_out vars

        if self.use_norm:
            future_pred = future_pred * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            future_pred = future_pred + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return future_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass implementing the complete MODE algorithm.

        Returns the predicted future time series Ŷ = (y_{L+1}, ..., y_{L+H})
        as specified in Step 15 of the pseudocode.
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]
