"""M3 modality masking for ACT policies trained on bimanual YAM data.

Adapts the Modality Masking Mechanism from "Robust Bimanual Vision-Language-Action
Models via Embarrassingly Simple Modality Masking" (arXiv 2608.22419) to the ACT
policy of YAMLab's LeRobot v2.0 fork. The masking is training-only: in eval mode the
model is exactly ACT, and checkpoints load into a stock ``ACTPolicy``.

What is masked, per training sample:

- Vision: the ego (top) camera is always visible. With probability
  ``wrist_mask_prob`` *both* wrist cameras are hidden together, never one alone.
  Hidden camera tokens are removed as attention keys in the transformer encoder and
  in the decoder cross-attention.
- Queries: each of the ``chunk_size`` decoder queries is hidden with probability
  ``query_mask_prob`` (never all of them). Hidden queries are removed as keys in the
  decoder self-attention, and the visible ones are rescaled by ``1 / (1 - p)``.
  This needs ``n_decoder_layers >= 2``: the fork's default of 1 has a decoder
  self-attention that outputs zeros, so there is nothing to hide.

The paper also masks language tokens. ACT takes no language input, so that part does
not apply here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import (
    ACT,
    ACTDecoder,
    ACTDecoderLayer,
    ACTEncoder,
    ACTPolicy,
)


@dataclass
class M3Config:
    # Camera that is never masked (the paper's egocentric spatial anchor).
    ego_camera: str = "observation.images.top_rgb"
    # Probability of hiding all wrist cameras together for a training sample.
    wrist_mask_prob: float = 0.3
    # Probability of hiding each decoder query for a training sample.
    query_mask_prob: float = 0.1
    # Rescale visible queries by 1 / (1 - query_mask_prob), as in the paper.
    query_rescale: bool = True

    def __post_init__(self):
        for name in ("wrist_mask_prob", "query_mask_prob"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {getattr(self, name)}.")


class _MaskState:
    """Masks for the current training forward pass, shared by the M3 encoder and decoder.

    Both fields are None outside a training forward pass, which makes the M3 modules
    behave exactly like their ACT parents.
    """

    def __init__(self, n_1d_tokens: int, wrist_cam_indices: list[int]):
        self.n_1d_tokens = n_1d_tokens
        self.wrist_cam_indices = wrist_cam_indices
        # Feature-map width of each camera, in camera order, recorded by a backbone hook.
        self.cam_widths: list[int] = []
        self.wrist_hidden: Tensor | None = None  # (B,) bool
        self.query_hidden: Tensor | None = None  # (B, chunk_size) bool

    def clear(self):
        self.cam_widths = []
        self.wrist_hidden = None
        self.query_hidden = None

    def encoder_token_mask(self, n_tokens: int) -> Tensor | None:
        """(B, n_tokens) key padding mask over encoder tokens; True hides a token."""
        if self.wrist_hidden is None or not self.wrist_hidden.any():
            return None
        # ACT concatenates the camera feature maps along the width and flattens them row by row,
        # so a token's camera is given by its column.
        total_width = sum(self.cam_widths)
        n_rows, remainder = divmod(n_tokens - self.n_1d_tokens, total_width)
        if remainder != 0 or n_rows == 0:
            raise RuntimeError(
                f"Cannot map {n_tokens} encoder tokens onto cameras of feature widths {self.cam_widths}."
            )
        device = self.wrist_hidden.device
        column_is_wrist = torch.zeros(total_width, dtype=torch.bool, device=device)
        start = 0
        for cam_index, width in enumerate(self.cam_widths):
            if cam_index in self.wrist_cam_indices:
                column_is_wrist[start : start + width] = True
            start += width
        token_is_wrist = torch.cat(
            [
                torch.zeros(self.n_1d_tokens, dtype=torch.bool, device=device),
                column_is_wrist.repeat(n_rows),
            ]
        )
        return self.wrist_hidden.unsqueeze(1) & token_is_wrist.unsqueeze(0)


class M3ACTEncoder(ACTEncoder):
    def __init__(self, config: ACTConfig, state: _MaskState):
        super().__init__(config)
        self._m3_state = state

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        if key_padding_mask is None:
            key_padding_mask = self._m3_state.encoder_token_mask(x.shape[0])
        return super().forward(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)


def _masked_decoder_layer(
    layer: ACTDecoderLayer,
    x: Tensor,
    encoder_out: Tensor,
    decoder_pos_embed: Tensor,
    encoder_pos_embed: Tensor,
    query_mask: Tensor | None,
    encoder_token_mask: Tensor | None,
) -> Tensor:
    """`ACTDecoderLayer.forward` with key padding masks on both attention blocks."""
    skip = x
    if layer.pre_norm:
        x = layer.norm1(x)
    q = k = layer.maybe_add_pos_embed(x, decoder_pos_embed)
    x = layer.self_attn(q, k, value=x, key_padding_mask=query_mask)[0]
    x = skip + layer.dropout1(x)
    if layer.pre_norm:
        skip = x
        x = layer.norm2(x)
    else:
        x = layer.norm1(x)
        skip = x
    x = layer.multihead_attn(
        query=layer.maybe_add_pos_embed(x, decoder_pos_embed),
        key=layer.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
        value=encoder_out,
        key_padding_mask=encoder_token_mask,
    )[0]
    x = skip + layer.dropout2(x)
    if layer.pre_norm:
        skip = x
        x = layer.norm3(x)
    else:
        x = layer.norm2(x)
        skip = x
    x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = skip + layer.dropout3(x)
    if not layer.pre_norm:
        x = layer.norm3(x)
    return x


class M3ACTDecoder(ACTDecoder):
    def __init__(self, config: ACTConfig, state: _MaskState, query_scale: float):
        super().__init__(config)
        self._m3_state = state
        self._query_scale = query_scale

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        state = self._m3_state
        encoder_token_mask = state.encoder_token_mask(encoder_out.shape[0])
        query_mask = state.query_hidden
        if query_mask is not None and not query_mask.any():
            query_mask = None
        if encoder_token_mask is None and query_mask is None:
            return super().forward(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )

        if query_mask is not None and self._query_scale != 1.0:
            # (DS, 1, C) query embeddings become per-sample (DS, B, C): visible queries are scaled up.
            scale = torch.where(query_mask, 1.0, self._query_scale).to(decoder_pos_embed.dtype)
            decoder_pos_embed = decoder_pos_embed * scale.transpose(0, 1).unsqueeze(-1)

        for layer in self.layers:
            x = _masked_decoder_layer(
                layer, x, encoder_out, decoder_pos_embed, encoder_pos_embed, query_mask, encoder_token_mask
            )
        return self.norm(x)


class M3ACT(ACT):
    """ACT whose encoder and decoder apply M3 masks during training. Same parameters as ACT."""

    def __init__(self, config: ACTConfig, m3: M3Config):
        super().__init__(config)
        self.m3 = m3

        cameras = list(config.image_features)
        if m3.ego_camera not in cameras:
            raise ValueError(f"M3 ego camera '{m3.ego_camera}' is not a policy image input: {cameras}.")
        wrist_cam_indices = [i for i, key in enumerate(cameras) if key != m3.ego_camera]
        if m3.wrist_mask_prob > 0 and not wrist_cam_indices:
            raise ValueError("M3 wrist masking needs at least one camera besides the ego camera.")
        if m3.query_mask_prob > 0 and config.n_decoder_layers < 2:
            # The decoder input is all zeros, so the first layer's self-attention outputs zeros whatever
            # its keys are. Hiding queries only takes effect from the second decoder layer on.
            logging.warning(
                "M3 query masking has no effect with n_decoder_layers=1 (only the query rescaling "
                "applies). Use --policy.n_decoder_layers=2 or more to mask queries."
            )

        self._m3_state = _MaskState(self.encoder_1d_feature_pos_embed.num_embeddings, wrist_cam_indices)
        query_scale = 1.0 / (1.0 - m3.query_mask_prob) if m3.query_rescale else 1.0
        self.encoder = M3ACTEncoder(config, self._m3_state)
        self.decoder = M3ACTDecoder(config, self._m3_state, query_scale)
        self._reset_parameters()
        self.backbone.register_forward_hook(self._record_cam_width)

    def _record_cam_width(self, module, inputs, output):
        self._m3_state.cam_widths.append(output["feature_map"].shape[-1])

    def _sample_masks(self, batch_size: int, device: torch.device):
        state = self._m3_state
        if self.m3.wrist_mask_prob > 0:
            state.wrist_hidden = torch.rand(batch_size, device=device) < self.m3.wrist_mask_prob
        if self.m3.query_mask_prob > 0:
            n_queries = self.config.chunk_size
            hidden = torch.rand(batch_size, n_queries, device=device) < self.m3.query_mask_prob
            # Never hide every query of a sample: bring one back at random.
            all_hidden = hidden.all(dim=1)
            keep = torch.randint(n_queries, (batch_size,), device=device)
            hidden[all_hidden, keep[all_hidden]] = False
            state.query_hidden = hidden

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        self._m3_state.clear()
        if not self.training:
            return super().forward(batch)
        images = batch["observation.images"]
        self._sample_masks(images.shape[0], images.device)
        try:
            return super().forward(batch)
        finally:
            self._m3_state.clear()


class M3ACTPolicy(ACTPolicy):
    """`ACTPolicy` trained with M3 masking.

    Keeps the name, config class and parameter names of `ACTPolicy`, so its checkpoints are
    ordinary ACT checkpoints. Set `M3ACTPolicy.m3` before the policy is built to change the
    masking settings.
    """

    m3: M3Config = M3Config()

    def __init__(self, config: ACTConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config, dataset_stats)
        self.model = M3ACT(config, self.m3)
