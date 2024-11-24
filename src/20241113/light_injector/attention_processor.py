
from typing import Callable, List, Optional, Tuple, Union

import torch 
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention

class LightJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("LightJointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        light_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:

            # encoder_hidden_states:  torch.Size([1, 333, 1536])

            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # encoder_hidden_states_query_proj:  torch.Size([1, 333, 1536])
            # encoder_hidden_states_key_proj:  torch.Size([1, 333, 1536])
            # encoder_hidden_states_value_proj:  torch.Size([1, 333, 1536])

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # `light` projections.
        if light_hidden_states is not None:
            # light_hidden_states:  torch.Size([1, 1536])
            light_hidden_states_query_proj = attn.light_q_proj(light_hidden_states)
            light_hidden_states_key_proj = attn.light_k_proj(light_hidden_states)
            light_hidden_states_value_proj = attn.light_v_proj(light_hidden_states)

            # light_hidden_states_query_proj:  torch.Size([1, 1536])
            # light_hidden_states_key_proj:  torch.Size([1, 1536])
            # light_hidden_states_value_proj:  torch.Size([1, 1536])

            light_hidden_states_query_proj = light_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            light_hidden_states_key_proj = light_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            light_hidden_states_value_proj = light_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # TODO: verify that actually need this norm
            if attn.norm_light_q is not None:
                light_hidden_states_query_proj = attn.norm_light_q(light_hidden_states_query_proj)
            if attn.norm_light_k is not None:
                light_hidden_states_key_proj = attn.norm_light_k(light_hidden_states_key_proj)

            query = torch.cat([query, light_hidden_states_query_proj], dim=2)
            key = torch.cat([key, light_hidden_states_key_proj], dim=2)
            value = torch.cat([value, light_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        # with_light: torch.Size([1, 24, 4430, 64])
        # without_light: torch.Size([1, 24,  4429, 64])

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if light_hidden_states is not None:
            # Split the light out from attention output where light is last few dimension
            hidden_states, light_hidden_states = (
                hidden_states[:, : -light_hidden_states.shape[1]],
                hidden_states[:, -light_hidden_states.shape[1] :]
            )

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if light_hidden_states is not None and encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states, light_hidden_states
        
        if light_hidden_states is not None:
            return hidden_states, light_hidden_states

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
