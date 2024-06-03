import torch 
from diffusers.models.resnet import ResnetBlock2D
from diffusers.utils import deprecate


class LightEmbedBlock(torch.nn.Module):

    def __init__(self, channel, *args, **kwargs):
        super().__init__()
        self.light_mul = torch.nn.Parameter(torch.ones(2, channel))
        self.light_add = torch.nn.Parameter(torch.zeros(2, channel))
        # spinkle random noise 
        self.light_add.data = self.light_add.data*torch.randn_like(self.light_add) * 1e-3


        self.light_direction = 0
    
    def set_light_direction(self, direction):
        self.light_direction = direction

    def forward(self, x):
        return x * self.light_mul[self.light_direction][None,:,None,None] + self.light_add[self.light_direction][None,:,None,None] 



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
            light_mul, light_add = self.get_lightcondition()
            
            hidden_states = light_mul*hidden_states + light_add + temb
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
        light_mul, light_add = self.get_lightcondition()    
        hidden_states = light_mul*hidden_states + light_add 
        hidden_states = self.norm2(hidden_states)

    hidden_states = self.nonlinearity(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
        input_tensor = self.conv_shortcut(input_tensor)

    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

    return output_tensor


# Inject to Unet
def set_light_direction(self, direction):
    for block_id in range(len(self.down_blocks)):
        for resblock_id in len(self.down_blocks[block_id].resblocks):
            if hasattr(self.down_blocks[block_id].resblocks[resblock_id], 'time_emb_proj'):
                self.down_blocks[block_id].resblocks[resblock_id].light_block.set_light_direction(direction)
    for block_id in range(self.up_blocks):
        for resblock_id in len(self.up_blocks[block_id].resblocks):
            if hasattr(self.up_blocks[block_id].resblocks[resblock_id], 'time_emb_proj'):
                self.up_blocks[block_id].resblocks[resblock_id].light_block.set_light_direction(direction)

def add_light_block(self):
    for block_id in range(3):
        for resblock_id in len(self.down_blocks[block_id].resblocks):
            num_channel = self.unet.down_blocks[block_id].resnets[resblock_id].time_emb_proj.out_features
            self.down_blocks[block_id].resblocks[resblock_id].light_block = LightEmbedBlock(num_channel)
            self.down_blocks[block_id].resblocks[resblock_id].forward = forward_lightcondition
    for block_id in range(3):
        for resblock_id in len(self.up_blocks[block_id].resblocks):
            num_channel = self.unet.up_blocks[block_id].resnets[resblock_id].time_emb_proj.out_features
            self.up_blocks[block_id].resblocks[resblock_id].light_block = LightEmbedBlock(num_channel)
            self.up_blocks[block_id].resblocks[resblock_id].forward = forward_lightcondition

def get_optimizable_params(self):
    all_params = []
    for block_id in range(3):
        for resblock_id in len(self.down_blocks[block_id].resblocks):
            all_params.extend(self.down_blocks[block_id].resblocks[resblock_id].light_block.parameters())
    for block_id in range(3):
        for resblock_id in len(self.up_blocks[block_id].resblocks):
            all_params.extend(self.up_blocks[block_id].resblocks[resblock_id].light_block.parameters())
    return all_params


