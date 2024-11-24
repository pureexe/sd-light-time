import torch 
from diffusers.models.resnet import ResnetBlock2D
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
            light_m = self.light_mul(direction) #1,1408
            light_a = self.light_add(direction) #1,1408, v.shape[1,704,64,64]
            y = v * light_m[...,None,None] + light_a[...,None,None]
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
        if temb is None:
            raise ValueError(
                f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
            )
        time_scale, time_shift = torch.chunk(temb, 2, dim=1)
        hidden_states = self.light_block(hidden_states) # similar to default, we add light block before norm2
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states * (1 + time_scale) + time_shift
    else:
        hidden_states = self.light_block(hidden_states)
        hidden_states = self.Affin(hidden_states)

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
            if hasattr(self.down_blocks[block_id].resnets[resblock_id], 'time_emb_proj') and hasattr(self.down_blocks[block_id].resnets[resblock_id], 'light_block'):
                self.down_blocks[block_id].resnets[resblock_id].light_block.set_light_direction(direction)
                self.down_blocks[block_id].resnets[resblock_id].light_block.set_apply_cfg(is_apply_cfg) 
    for block_id in range(len(self.up_blocks)):
        for resblock_id in range(len(self.up_blocks[block_id].resnets)):
            if hasattr(self.up_blocks[block_id].resnets[resblock_id], 'time_emb_proj') and hasattr(self.up_blocks[block_id].resnets[resblock_id], 'light_block'):
                self.up_blocks[block_id].resnets[resblock_id].light_block.set_light_direction(direction)
                self.up_blocks[block_id].resnets[resblock_id].light_block.set_apply_cfg(is_apply_cfg)


def add_light_block(self, in_channel=3):
    for block_id in range(len(self.down_blocks)):
        for resblock_id in range(len(self.down_blocks[block_id].resnets)):
            num_channel = self.down_blocks[block_id].resnets[resblock_id].time_emb_proj.out_features
            if self.down_blocks[block_id].resnets[resblock_id].time_embedding_norm == "scale_shift":
                num_channel = num_channel // 2 #support scale_shift mode
            self.down_blocks[block_id].resnets[resblock_id].light_block = LightEmbedBlock(num_channel, in_dim=in_channel)
            self.down_blocks[block_id].resnets[resblock_id].forward = types.MethodType(forward_lightcondition, self.down_blocks[block_id].resnets[resblock_id])

    for block_id in range(len(self.up_blocks)):
        for resblock_id in range(len(self.up_blocks[block_id].resnets)):
            num_channel = self.up_blocks[block_id].resnets[resblock_id].time_emb_proj.out_features
            if self.up_blocks[block_id].resnets[resblock_id].time_embedding_norm == "scale_shift":
                num_channel = num_channel // 2 #support scale_shift mode
            self.up_blocks[block_id].resnets[resblock_id].light_block = LightEmbedBlock(num_channel, in_dim=in_channel)
            self.up_blocks[block_id].resnets[resblock_id].forward = types.MethodType(forward_lightcondition, self.up_blocks[block_id].resnets[resblock_id])

# Initialize weights and biases to ensure the output is always 0
def initialize_weights_of_last_output(model, target = 0):
    if not isinstance(model[-1], torch.nn.Linear):
        raise ValueError("The model should end with a linear layer to initialize the weights and biases.")
    model[-1].weight.data.fill_(0.0)  # Set weights to zero
    model[-1].bias.data.fill_(target)    # Set biases to zero
    return model
