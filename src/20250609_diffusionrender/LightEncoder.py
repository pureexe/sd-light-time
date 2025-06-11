"""
Config from light encoder
"""
import torch
from typing import Optional, Tuple, Union
from diffusers.configuration_utils  import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import get_down_block
from diffusers.models.unets.unet_2d import UNet2DOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps

class LightEncoder(ModelMixin, ConfigMixin):
    """
    Light encoder that taken VAE(E_LDR), VAE(LOG_HDR), and (E_DIR) as input that produce output for each sequence
    """
    @register_to_config
    def __init__(self, 
            in_channels: int = 12,
            block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
            down_block_types: Tuple[str, ...] = (
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D"
            ),
            act_fn: str='silu',
            cross_attention_dim: int = 768,
            layers_per_block: int = 2,
            time_embedding_type: str = "positional",
            time_embedding_dim: Optional[int] = None,
            freq_shift: int = 0,
            flip_sin_to_cos: bool = True,
            norm_eps: float = 1e-5,
            norm_num_groups: int = 32,
            attention_head_dim: Optional[int] = 8,
            downsample_padding: int = 1,
            resnet_time_scale_shift: str = "default",
            downsample_type: str = "conv",
            dropout: float = 0.0,
            class_embed_type: Optional[str] = None,
            num_class_embeds: Optional[int] = None,
            num_train_timesteps: Optional[int] = None,
        ):
        super().__init__()

        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        # input
        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == "learned":
            self.time_proj = torch.nn.Embedding(num_train_timesteps, block_out_channels[0])
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = torch.nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = torch.nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None


        self.down_blocks = torch.nn.ModuleList([])
        self.feature_to_cross_attention = torch.nn.ModuleList([])

        output_channel = block_out_channels[0]
        
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)
            # PURE: We map feature to cross attention dimension here
            self.feature_to_cross_attention.append(
                torch.nn.Conv2d(output_channel, cross_attention_dim, kernel_size=1, stride=1, padding=0)
            )

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            class_labels: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        
        # 0. center input if necessary
        # if self.config.center_input_sample:
        #     sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

         # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        down_output_samples = []
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

            output_sample = sample
            if self.config.time_embedding_type == "fourier":
                timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
                output_sample = output_sample / timesteps
            down_output_samples.append(output_sample)

        # convert to cross attention dimension
        output_feature = []
        for i, sample in enumerate(down_output_samples):
            n_feature = self.feature_to_cross_attention[i](sample)
            # flattenting the spatial dimensions
            n_feature = n_feature.flatten(2).transpose(1, 2) #[batch, seq_len, feature_dim]
            output_feature.append(
                n_feature
            )

        output_feature = torch.cat(output_feature, dim=1)  # [batch, seq_len, feature_dim]

        if not return_dict:
            return output_feature

        return UNet2DOutput(
            sample=output_feature
        )




        
