"""Hooked Transformer Embed Component.

This module contains all the component :class:`Embed`.
"""
from typing import Dict, List, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int
# from transformer_lens.components import LayerNorm
from .layer_norm import LayerNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig],

                 # not official implementation
                 vision_tower=None,
                 multi_modal_projector=None):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=self.cfg.dtype)
        )

        # not official implementation
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

        # Some models (e.g. Bloom) need post embedding layer norm
        if self.cfg.post_embedding_ln:
            self.ln = LayerNorm(self.cfg)

    # not official implementation
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.cfg.vision_feature_select_strategy}")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    def forward(
        self, tokens: Int[torch.Tensor, "batch pos"],

            # not official implememntation
            pixel_values: Float[torch.Tensor, "batch n_images"] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)

        emb = self.W_E[tokens, :]
        if self.cfg.post_embedding_ln:
            return self.ln(emb)

        if "llava" not in self.cfg.original_architecture.lower():
            return emb
        else:
            vision_feature_layer = self.cfg.vision_feature_layer
            vision_feature_select_strategy = self.cfg.vision_feature_select_strategy

            inputs_embeds = emb

            if pixel_values is not None:
                image_features = self.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

                n_image_tokens = (
                            tokens == self.cfg.image_token_index).sum().item()
                n_image_features = image_features.shape[0] * \
                                   image_features.shape[1]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                special_image_mask = (
                            tokens == self.cfg.image_token_index).unsqueeze(
                    -1)
                special_image_mask = special_image_mask.expand_as(
                    inputs_embeds).to(inputs_embeds.device)
                image_features = image_features.to(inputs_embeds.device,
                                                   inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_image_mask, image_features)

            return inputs_embeds
