from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from ..ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from ..ops.triton.selective_state_update import selective_state_update
from ..ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


class Mamba(nn.Module):
    """
    MODE: The enhanced Mamba module with low-rank Neural ODE option
    """

    def _make_bc_proj(self, **factory_kwargs):
        """
        B(t) and C(t) generators: depth-wise conv (sequence mixing) + point-wise conv → d_state
        """
        return nn.Sequential(
            # Depth-wise convolution along sequence dimension
            nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=2,
                padding=1,
                groups=self.d_inner,  # depth-wise
                bias=False,
                **factory_kwargs,
            ),
            nn.SiLU(),
            # Point-wise conv mixes channels and projects to d_state
            nn.Conv1d(
                self.d_inner,
                self.d_state,
                kernel_size=1,
                bias=True,
                **factory_kwargs,
            ),
        )

    def _make_uv_proj(self, **factory_kwargs):
        """
        Convolutional generators for U(t) and V(t)
        """
        return nn.Sequential(
            nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=2,  # matches B/C depth-wise conv setting
                padding=1,
                groups=self.d_inner,
                bias=False,
                **factory_kwargs,
            ),
            nn.SiLU(),
            nn.Conv1d(
                self.d_inner,
                self.d_inner * self.r_rank,
                kernel_size=1,
                bias=False,
                **factory_kwargs,
            ),
        )

    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            low_rank_ode=True,
            r_rank=8,
            static_low_rank=True,  # if True, use constant A=D+UV^T instead of time-varying A(t)
            hippo_init=False,  # if True and static_low_rank, initialize U,V via HiPPO SVD
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        if low_rank_ode:
            # Disable fast fused path since it doesn't support low-rank time-varying A
            self.use_fast_path = False
        self.layer_idx = layer_idx

        # Low-rank Neural ODE switch
        self.low_rank_ode = low_rank_ode
        self.r_rank = r_rank if low_rank_ode else None

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

        self.b_proj = self._make_bc_proj(**factory_kwargs)
        self.c_proj = self._make_bc_proj(**factory_kwargs)

        # When using low-rank ODE, we need projections that generate U(t), V(t) and C(t)
        if self.low_rank_ode:
            self.u_proj = self._make_uv_proj(**factory_kwargs)
            self.v_proj = self._make_uv_proj(**factory_kwargs)

        self.use_static_low_rank = static_low_rank and low_rank_ode
        if self.use_static_low_rank:
            # Learnable low-rank factors producing global correction UVᵀ (rank=r_rank)
            self.U_param = nn.Parameter(torch.randn(self.d_inner, self.r_rank) * 0.02)
            self.V_param = nn.Parameter(torch.randn(self.r_rank, self.d_state) * 0.02)
            self.low_rank_scale = nn.Parameter(torch.tensor(0.05))

            # Optional HiPPO initialization
            if hippo_init:
                with torch.no_grad():
                    # Build HiPPO full matrix (size d_state x d_state)
                    i_idx = torch.arange(
                        self.d_state, device=device, dtype=torch.float64
                    ).unsqueeze(1)
                    j_idx = torch.arange(
                        self.d_state, device=device, dtype=torch.float64
                    ).unsqueeze(0)

                    # Create matrix with numerical stability
                    hippo = torch.zeros(self.d_state, self.d_state, device=device, dtype=torch.float64)

                    # Fill diagonal with 1s
                    hippo.fill_diagonal_(1.0)

                    # Fill off-diagonal elements with the HiPPO formula
                    for i in range(self.d_state):
                        for j in range(self.d_state):
                            if i != j:
                                # Correct HiPPO LegS formula: A_{HiPPO} = - (i+j+1) / (i+j+2)
                                # This formula is numerically stable (denominator always >= 2 for i,j >= 0)
                                denominator = (i + j + 2.0)
                                hippo[i, j] = -(i + j + 1.0) / denominator

                    # # Scale the matrix to improve conditioning
                    # hippo = hippo / torch.norm(hippo, dim=1, keepdim=True).clamp(min=1.0)

                    # SVD with better numerical stability
                    try:
                        U_svd, S_svd, Vh_svd = torch.linalg.svd(hippo, full_matrices=False)
                    except torch._C._LinAlgError:
                        # Fallback: use eigendecomposition for ill-conditioned matrices
                        eigenvals, eigenvecs = torch.linalg.eig(hippo)
                        S_svd = torch.sqrt(torch.abs(eigenvals.real))
                        U_svd = eigenvecs.real
                        Vh_svd = eigenvecs.inverse().real

                    # Convert back to float32 for compatibility
                    U_svd = U_svd.to(torch.float32)
                    S_svd = S_svd.to(torch.float32)
                    Vh_svd = Vh_svd.to(torch.float32)

                    r = self.r_rank
                    U_r = U_svd[:, :r] * torch.sqrt(S_svd[:r]).unsqueeze(
                        0
                    )  # (d_state, r)
                    V_r = (torch.sqrt(S_svd[:r]).unsqueeze(1) * Vh_svd[:r, :]).to(
                        device
                    )  # (r, d_state)

                    # Initialize V_param directly
                    self.V_param.copy_(V_r.to(self.V_param.dtype))

                    # For U_param (d_inner x r), tile rows cyclically from U_r
                    repeats = (self.d_inner + self.d_state - 1) // self.d_state
                    tiled_U = U_r.repeat(repeats, 1)[: self.d_inner]
                    self.U_param.copy_(tiled_U.to(self.U_param.dtype))

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        if getattr(self, "use_static_low_rank", False):
            A = A + self.low_rank_scale * (
                torch.matmul(self.U_param, self.V_param) / math.sqrt(self.r_rank)
            )
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
                self.use_fast_path
                and causal_conv1d_fn is not None
                and inference_params is None
        ):  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(x, (self.d_conv - x.shape[-1], 0))
                )  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            assert self.activation in ["silu", "swish"]
            # Branch: static low-rank vs dynamic segmented
            if self.low_rank_ode and getattr(self, "use_static_low_rank", False):
                # Constant A = D + α·UVᵀ
                A_const = A + self.low_rank_scale * (
                    torch.matmul(self.U_param, self.V_param) / math.sqrt(self.r_rank)
                )

                y = selective_scan_fn(
                    x,
                    dt,
                    A_const,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b d l -> b l d")
                out = self.out_proj(y)
            elif self.low_rank_ode:
                # Low-rank Neural ODE with segmented selective scan
                U_all = self.u_proj(x)[..., :seqlen]  # ensure same seq len
                V_all = self.v_proj(x)[..., :seqlen]

                # Reshape (b, d_inner*r, l) -> (b, d_inner, r, l)
                U_all = U_all.view(batch, self.d_inner, self.r_rank, seqlen)
                V_all = V_all.view(batch, self.d_inner, self.r_rank, seqlen)

                # Recompute B and C using conv-based generators.
                if x.dim() == 2:
                    _inp = x.unsqueeze(-1)
                else:
                    _inp = x

                B_raw = self.b_proj(_inp)[..., :seqlen]  # ensure same seq len
                C_raw = self.c_proj(_inp)[..., :seqlen]

                if B_raw.ndim == 3 and B_raw.shape[-1] == 1:
                    B = B_raw[..., 0]  # (B, d_state)
                    C = C_raw[..., 0]
                else:
                    B = B_raw  # (B, d_state, L)
                    C = C_raw

                segment_len = getattr(self, "segment_len", 1)
                ys = []
                state = None

                for start in range(0, seqlen, segment_len):
                    end = min(start + segment_len, seqlen)
                    seg_len = end - start

                    # Use first timestep's U,V as constant for the block
                    U_seg = U_all[..., start]  # (b, d_inner, r)
                    V_seg = V_all[..., start]  # (b, d_inner, r)

                    # We collapse low-rank factors: A = U (elementwise) V^T with r columns treated as d_state
                    # A_seg = torch.einsum('b d r, b d r -> b d r', U_seg, V_seg) / math.sqrt(self.r_rank)
                    A_raw = U_seg * V_seg
                    A_seg = -0.1 * torch.tanh(A_raw) / math.sqrt(float(self.r_rank))
                    # A_seg shape (B, d_inner, r_rank)
                    A_const = A_seg.mean(dim=0)  # (d_inner, r_rank)

                    if A_const.shape[1] > B.shape[1]:
                        A_const = A_const[:, : B.shape[1]].contiguous()
                    elif A_const.shape[1] < B.shape[1]:
                        # Properly repeat A_const columns to pad to match B.shape[1]
                        pad = B.shape[1] - A_const.shape[1]
                        # Calculate how many full repeats we need plus remainder
                        full_repeats = pad // A_const.shape[1]
                        remainder = pad % A_const.shape[1]
                        # Build padding tensor
                        padding_parts = [A_const] * (full_repeats + 1)  # +1 for the original
                        if remainder > 0:
                            padding_parts.append(A_const[:, :remainder])
                        # Concatenate all parts
                        A_const = torch.cat(padding_parts, dim=1)

                    # Prepare inputs for scan
                    x_seg = x[:, :, start:end]
                    dt_seg = dt[:, :, start:end]
                    B_seg = B[:, :, start:end].contiguous()  # (b, d_state, seg_len)
                    C_seg = C[:, :, start:end].contiguous()  # (b, d_state, seg_len)

                    # # Debug print for first segment
                    # if start == 0:
                    #     print(f"[DEBUG] x_seg shape: {x_seg.shape}, dt_seg shape: {dt_seg.shape}")
                    #     print(f"[DEBUG] B_seg shape: {B_seg.shape}, C_seg shape: {C_seg.shape}")
                    #     print(f"[DEBUG] A_const shape: {A_const.shape}")
                    #     print(f"[DEBUG] D shape: {self.D.float().shape}")
                    #     if z is not None:
                    #         print(f"[DEBUG] z shape: {z[:, :, start:end].shape}")

                    y_seg, state = selective_scan_fn(
                        x_seg,
                        dt_seg,
                        A_const,
                        B_seg,
                        C_seg,
                        self.D.float(),
                        z=z[:, :, start:end] if z is not None else None,
                        delta_bias=self.dt_proj.bias.float(),
                        delta_softplus=True,
                        return_last_state=True,
                    )
                    ys.append(rearrange(y_seg, "b d l -> b l d"))

                y = torch.cat(ys, dim=1)  # (b, L, d_inner)
                out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        if self.low_rank_ode:
            # Conv-based B/C expect sequence dim. Use last element when len>1.
            B = self.b_proj(x.unsqueeze(-1))[..., 0]  # (B, d_state)
            C = self.c_proj(x.unsqueeze(-1))[..., 0]

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
            self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self,
            hidden_states: Tensor,
            residual: Optional[Tensor] = None,
            inference_params=None,
    ):
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
