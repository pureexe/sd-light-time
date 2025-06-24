"""
a helper file for inject light function
"""
import torch 
from typing import Callable, List, Optional, Tuple, Union

from diffusers.models.attention_processor import Attention, AttnProcessor2_0

class LightAttentionProcessor(AttnProcessor2_0):
    def __init__(self,
            processor_id = 0,
            block_depth: int = 0,
            token_lengths: List[int] = [4096, 1024, 256, 64],
            text_token_length: int = 77,
            metadata_token_length: int = 1,
        ):
        super().__init__()
        self.processor_id = processor_id
        self.block_depth = block_depth 
        self.token_lengths = token_lengths
        self.text_token_length = text_token_length
        self.metadata_token_length = metadata_token_length


    def get_token(self, token):
        start_token = self.text_token_length + self.metadata_token_length
        for i in range(self.block_depth):
            start_token += self.token_lengths[i]
        end_token = start_token + self.token_lengths[self.block_depth]
        if len(token.shape) == 2:
            return token[start_token:end_token, :]
        else:
            return token[..., start_token:end_token, :]
        
    def get_text_token(self, token):
        if len(token.shape) == 2:
            return token[:self.text_token_length, :]
        else:
            return token[..., :self.text_token_length, :]
        
    def get_text_gate(self, token):
        meta_token_id = self.text_token_length 
        if len(token.shape) == 2:
            return token[meta_token_id, self.processor_id * 2]
        else:
            return token[..., meta_token_id, self.processor_id * 2]
        

    def get_light_gate(self, token):
        meta_token_id = self.text_token_length
        if len(token.shape) == 2:
            gate_token = token[meta_token_id]
            return gate_token[self.processor_id * 2 + 1] # shape 1
        else:
            gate_token = token[..., meta_token_id, :] # shape [B, 1]
            return gate_token[..., self.processor_id * 2 + 1]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)

        # encoder_hidden_states with be shape of [77 + 1 + 4096 + 1024 + 256, 768]

        text_embedings = self.get_text_token(encoder_hidden_states)
        text_output = super().__call__(attn, hidden_states, text_embedings, attention_mask, temb, *args, **kwargs)

        light_embedings = self.get_token(encoder_hidden_states)
        light_output = super().__call__(attn, hidden_states, light_embedings, attention_mask, temb, *args, **kwargs)

        text_gate = self.get_text_gate(encoder_hidden_states) 
        light_gate =  self.get_light_gate(encoder_hidden_states) 


        # expand shape
        num_token, num_channels = text_output.shape[1:]
        text_gate = text_gate[..., None, None].expand(-1, num_token,num_channels)
        light_gate = light_gate[..., None, None].expand(-1, num_token,num_channels)

        output = text_gate * text_output + light_gate * light_output
        return output


