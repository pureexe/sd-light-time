"""
a helper file for inject light function
"""
import torch 
from typing import Callable, List, Optional, Tuple, Union

from diffusers.models.attention_processor import Attention, AttnProcessor2_0

class LightAttentionProcessor(AttnProcessor2_0):
    def __init__(self, block_depth: int = 0, token_lengths: List[int] = [256, 64, 16, 16]):
        super().__init__()
        self.block_depth = 0
        self.token_lengths = token_lengths

    def get_token(self, token):
        start_token = 0
        for i in range(self.block_depth):
            start_token += self.token_lengths[i]
        end_token = start_token + self.token_lengths[self.block_depth]
        if len(token.shape) == 2:
            return token[start_token:end_token, :]
        else:
            return token[..., start_token:end_token, :]

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
        encoder_hidden_states = self.get_token(encoder_hidden_states) if encoder_hidden_states is not None else None
        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
