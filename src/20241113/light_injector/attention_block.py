
import torch
from torch import nn 
import types

from diffusers.models.attention import JointTransformerBlock, _chunked_feed_forward, FeedForward
from diffusers.models.normalization import  AdaLayerNormContinuous, AdaLayerNormZero, FP32LayerNorm, RMSNorm


def inject_block(
        block: JointTransformerBlock,
        dim: int
    ):
    eps = 1e-6
    #module for entire block
    if isinstance(block.norm1_context, AdaLayerNormContinuous):
        block.norm1_light = AdaLayerNormContinuous(
            dim, dim, elementwise_affine=False, eps=eps, bias=True, norm_type="layer_norm"
        )
    elif isinstance(block.norm1_context, AdaLayerNormZero):
        block.norm1_light = AdaLayerNormZero(dim)
    else:
        raise ValueError(
            f"inject_block currently only support `ada_norm_continous`, `ada_norm_zero`"
        )
    if not block.context_pre_only:
        block.norm2_light = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        block.ff_light = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
    else:
        block.norm2_light = None
        block.ff_light = None

    # attention module update
    for attn_id in ['attn', 'attn2']:
        # get attention from block
        if not hasattr(block, attn_id):
            continue
        attn = getattr(block, attn_id)
        if attn is None:
            continue    
        
        # add k,v,q projection to light with samesetting as a context (prompt)
        if attn.added_kv_proj_dim is not None:
            # register light k,q,v mlp
            attn.light_k_proj = nn.Linear(attn.added_kv_proj_dim, attn.inner_kv_dim, bias=attn.added_proj_bias)
            attn.light_v_proj = nn.Linear(attn.added_kv_proj_dim, attn.inner_kv_dim, bias=attn.added_proj_bias)
            if attn.context_pre_only is not None:
                attn.light_q_proj = nn.Linear(attn.added_kv_proj_dim, attn.inner_dim, bias=attn.added_proj_bias)

            # register norm layer for light
            if hasattr(attn, 'norm_added_q') and attn.norm_added_q is not None:
                norm_qk = attn.norm_added_q
                if hasattr(norm_qk, 'dim'):
                    norm_dim_head = norm_qk.dim 
                elif hasattr(norm_qk, 'normalized_shape'):
                    norm_dim_head = norm_qk.normalized_shape
                else:
                    raise ValueError("Light: cannot find dim_head inside norm_added_q")

                if isinstance(norm_qk, FP32LayerNorm):
                    attn.norm_light_q = FP32LayerNorm(norm_dim_head, elementwise_affine=False, bias=False, eps=eps)
                    attn.norm_light_k = FP32LayerNorm(norm_dim_head, elementwise_affine=False, bias=False, eps=eps)
                elif isinstance(norm_qk, RMSNorm):
                    attn.norm_light_q = RMSNorm(norm_dim_head, eps=eps)
                    attn.norm_light_k = RMSNorm(norm_dim_head, eps=eps)
                else:
                    raise ValueError("Light: unknown qk_norm.  Should be one of 'fp32_layer_norm','rms_norm'`")






    # self.added_proj_bias = added_proj_bias
    # if self.added_kv_proj_dim is not None:
    #     self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
    #     self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
    #     if self.context_pre_only is not None:
    #         self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)


    block.forward = types.MethodType(block_forward, block)


def block_forward(
        self, 
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        light_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor
    ):
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            norm_light_hidden_states = self.norm1_light(light_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

            norm_light_hidden_states, l_gate_msa, l_shift_mlp, l_scale_mlp, l_gate_mlp = self.norm1_light(
                light_hidden_states, emb=temb
            )
        # Attention.
        attn_output, context_attn_output, light_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, light_hidden_states=light_hidden_states
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            light_attn_output = l_gate_msa.unsqueeze(1) *  light_attn_output
            light_hidden_states = light_hidden_states + light_attn_output

            norm_light_hidden_states = self.norm2_light(light_hidden_states)
            norm_light_hidden_states = norm_light_hidden_states * (1 + l_scale_mlp[:, None]) + l_shift_mlp[:, None]

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
                light_ff_output = _chunked_feed_forward(
                    self.ff_light, norm_light_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
                light_ff_output = self.ff_light(norm_light_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            light_hidden_states = light_hidden_states + l_gate_mlp.unsqueeze(1) * light_ff_output

        return encoder_hidden_states, light_hidden_states, hidden_states

