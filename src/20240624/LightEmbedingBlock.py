import torch 
from diffusers.models.resnet import ResnetBlock2D
from diffusers.utils import deprecate
import types
import numpy as np


class LightEmbedBlock(torch.nn.Module):

    def __init__(self, out_dim, in_dim=1, hidden_layers=2, hidden_dim=256, posenc=None, *args, **kwargs):
        super().__init__()
        # possible posenc [nerf:10]
        self.light_direction = torch.zeros(in_dim)
        self.pos_enc = None
        self.in_dim = in_dim
        if posenc is not None and posenc.startswith("nerf"):
            self.pos_enc = "nerf"
            self.pe_level = int(posenc.split(":")[1])
            in_dim = (in_dim * 2) * self.pe_level
        
        self.light_mul = get_mlp(in_dim, hidden_dim,  hidden_layers, out_dim)
        self.light_add = get_mlp(in_dim, hidden_dim, hidden_layers,out_dim)
        self.gate = torch.nn.Parameter(torch.zeros(1))
        self.is_apply_cfg = False
    
    def enable_apply_cfg(self):
        self.is_apply_cfg = True
        
    def disable_apply_cfg(self):
        self.is_apply_cfg = False
        
    def set_apply_cfg(self, is_apply_cfg):
        self.is_apply_cfg = is_apply_cfg
    
    def set_light_direction(self, direction):
        assert direction.shape[-1] == self.in_dim
        self.light_direction = direction

    def get_direction_feature(self):
        # apply pos enc
        if self.pos_enc == "nerf":
            assert direction.shape[0] == 1 # currently support only single batch training for PE.
            sin_branh = torch.sin(2.0**torch.arange(self.pe_level) * np.pi * self.light_direction[None])
            cos_branh = torch.cos(2.0**torch.arange(self.pe_level) * np.pi * self.light_direction[None])
            direction = torch.cat([sin_branh.flatten(), cos_branh.flatten()], dim=0)
        else:
            direction = self.light_direction
        return direction


    def forward(self, x):
        # if using cfg (classifier guidance-free), only apply light condition to non cfg part
        use_cfg = self.is_apply_cfg and x.shape[0] % 2 == 0
        if use_cfg:
            # apply only non cfg part
            h = x.shape[0] // 2
            v = x[h:]
        else:
            v = x #[B,C,H,W]
        
        direction = self.get_direction_feature().to(v.device).to(v.dtype) #[B,C]

        light_m = self.light_mul(direction) 
        light_a = self.light_add(direction)
        
        # compute light condition
        
        adagn = v * light_m[...,None,None] + light_a[...,None,None]
        
        y = v + (self.gate * adagn)

        #  concat part that not apply light condition back
        if use_cfg:
            y = torch.cat([x[:h], y], dim=0)
            

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

# Inject to ResnetBlock2D
def forward_lightcondition(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs):
    if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
    
    hidden_states = input_tensor
    hidden_states = self.norm1(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)
    
    if self.upsample is not None:
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            input_tensor = input_tensor.contiguous()
            hidden_states = hidden_states.contiguous()
        input_tensor = self.upsample(input_tensor)
        hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
        input_tensor = self.downsample(input_tensor)
        hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if self.time_emb_proj is not None:
        if not self.skip_time_act:
            temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]

    if self.time_embedding_norm == "default":
        if temb is not None:
            hidden_states = self.light_block(hidden_states  + temb)
        hidden_states = self.norm2(hidden_states)
    elif self.time_embedding_norm == "scale_shift":
        raise NotImplementedError("Not support scale-shift mode yet.")
        if temb is None:
            raise ValueError(
                f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
            )
        time_scale, time_shift = torch.chunk(temb, 2, dim=1)
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states * (1 + time_scale) + time_shift
    else:
        hidden_states = self.light_block(hidden_states)
        hidden_states = self. Affin(hidden_states)

    hidden_states = self.nonlinearity(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
        input_tensor = self.conv_shortcut(input_tensor)

    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

    return output_tensor


# Inject to Unet
def set_light_direction(self, direction, is_apply_cfg=False):
    for block_id in range(len(self.down_blocks)):
        for resblock_id in range(len(self.down_blocks[block_id].resnets)):
            if hasattr(self.down_blocks[block_id].resnets[resblock_id], 'time_emb_proj'):
                self.down_blocks[block_id].resnets[resblock_id].light_block.set_light_direction(direction)
                self.down_blocks[block_id].resnets[resblock_id].light_block.set_apply_cfg(is_apply_cfg) 
    for block_id in range(len(self.up_blocks)):
        for resblock_id in range(len(self.up_blocks[block_id].resnets)):
            if hasattr(self.up_blocks[block_id].resnets[resblock_id], 'time_emb_proj'):
                self.up_blocks[block_id].resnets[resblock_id].light_block.set_light_direction(direction)
                self.up_blocks[block_id].resnets[resblock_id].light_block.set_apply_cfg(is_apply_cfg) 

def add_light_block(self):
    for block_id in range(len(self.down_blocks)):
        for resblock_id in range(len(self.down_blocks[block_id].resnets)):
            num_channel = self.down_blocks[block_id].resnets[resblock_id].time_emb_proj.out_features
            self.down_blocks[block_id].resnets[resblock_id].light_block = LightEmbedBlock(num_channel)
            self.down_blocks[block_id].resnets[resblock_id].forward = types.MethodType(forward_lightcondition, self.down_blocks[block_id].resnets[resblock_id])

    for block_id in range(len(self.up_blocks)):
        for resblock_id in range(len(self.up_blocks[block_id].resnets)):
            num_channel = self.up_blocks[block_id].resnets[resblock_id].time_emb_proj.out_features
            self.up_blocks[block_id].resnets[resblock_id].light_block = LightEmbedBlock(num_channel)
            self.up_blocks[block_id].resnets[resblock_id].forward = types.MethodType(forward_lightcondition, self.up_blocks[block_id].resnets[resblock_id])


