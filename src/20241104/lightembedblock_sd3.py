import torch 
from diffusers.models.attention import _chunked_feed_forward, JointTransformerBlock
from diffusers.utils import deprecate
import types
import numpy as np


class LightEmbedBlock(torch.nn.Module):

    def __init__(self, out_dim, in_dim=1, hidden_layers=2, hidden_dim=256, *args, **kwargs):
        super().__init__()

        self.light_direction = torch.zeros(in_dim)
        self.in_dim = in_dim
        
        self.light_mul = get_mlp(in_dim, hidden_dim,  hidden_layers, out_dim)
        self.light_add = get_mlp(in_dim, hidden_dim, hidden_layers,out_dim)
        # set default weight  
        self.light_mul = initialize_weights_of_last_output(self.light_mul, target = 1)
        self.light_add = initialize_weights_of_last_output(self.light_add, target = 0)
        
        self.is_apply_cfg = False


    def enable_apply_cfg(self):
        self.is_apply_cfg = True
        
    def disable_apply_cfg(self):
        self.is_apply_cfg = False
        
    def set_apply_cfg(self, is_apply_cfg):
        self.is_apply_cfg = is_apply_cfg
    
    def set_light_direction(self, direction):
        assert direction == None or direction.shape[-1] == self.in_dim
        self.light_direction = direction

    def get_direction_feature(self):
        direction = self.light_direction
        return direction


    def forward(self, x):
        # if using cfg (classifier guidance-free), only apply light condition to non cfg part
        use_cfg = self.is_apply_cfg and x.shape[0] % 2 == 0
        x_uncond = None
        if use_cfg:
            # apply only non cfg part
            x_uncond, v = torch.chunk(x, 2)
        else:
            v = x #[B,C,H,W]
        
        direction = self.get_direction_feature()
        
        # apply light direction when set
        if direction is not None:
            direction = direction.to(v.device).to(v.dtype) #[B,C]
            light_m = self.light_mul(direction)
            light_a = self.light_add(direction)
            y = v * light_m[None, :] + light_a[None, :]
        else:
            y = v

        #  concat part that not apply light condition back
        if use_cfg:
            y = torch.cat([x_uncond, y], dim=0)

        return y

def get_mlp(in_dim, hidden_dim, hidden_layers, out_dim):
    # generate some classic MLP
    layers = []
    layers.append(torch.nn.Linear(in_dim, hidden_dim))  # input layer
    
    # Hidden layers
    for _ in range(hidden_layers):
        layers.append(torch.nn.ReLU())  # activation function
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))  # hidden layer
    
    layers.append(torch.nn.ReLU())  # activation function
    layers.append(torch.nn.Linear(hidden_dim, out_dim))  # output layer

    mlp = torch.nn.Sequential(*layers)
    
    
    return mlp

# Inject to JointTransformerBlock
def forward_lightcondition(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
    ):
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        # inject lighting condition here 
        hidden_states = self.light_block(hidden_states)

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
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states




# Inject to Unet
def set_light_direction(transformer, direction, is_apply_cfg=False):
    num_blocks = len(transformer.transformer_blocks)
    for block_id in range(num_blocks):
        transformer.transformer_blocks[block_id].light_block.set_light_direction(direction)
        transformer.transformer_blocks[block_id].light_block.set_apply_cfg(is_apply_cfg) 

def add_light_block(transformer, in_channel=3):
    num_blocks = len(transformer.transformer_blocks)
    for block_id in range(num_blocks):
        out_channel = transformer.transformer_blocks[block_id].norm1.linear.in_features
        transformer.transformer_blocks[block_id].light_block = LightEmbedBlock(out_channel, in_dim=in_channel)
        transformer.transformer_blocks[block_id].forward = types.MethodType(forward_lightcondition, transformer.transformer_blocks[block_id])


# Initialize weights and biases to ensure the output is always 0
def initialize_weights_of_last_output(model, target = 0):
    if not isinstance(model[-1], torch.nn.Linear):
        raise ValueError("The model should end with a linear layer to initialize the weights and biases.")
    model[-1].weight.data.fill_(0.0)  # Set weights to zero
    model[-1].bias.data.fill_(target)    # Set biases to zero
    return model
