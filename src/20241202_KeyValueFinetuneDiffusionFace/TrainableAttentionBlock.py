import torch

def enable_grad_on_kv(unet):
    """
    enable require grad on all k and v to make k,v trianble 
    Note that attn2 is for cross attention
    """
    # DOWN_BLOCK
    for block_id in range(len(unet.down_blocks)):
        if not hasattr(unet.down_blocks[block_id], 'attentions'):
            continue
        for attention_id in range(len(unet.down_blocks[block_id].attentions)):
            for transformer_id in range(len(unet.down_blocks[block_id].attentions[attention_id].transformer_blocks)):
                unet.down_blocks[block_id].attentions[attention_id].transformer_blocks[transformer_id].attn2.to_k.requires_grad_(True)
                unet.down_blocks[block_id].attentions[attention_id].transformer_blocks[transformer_id].attn2.to_v.requires_grad_(True)
    #UP_BLOCK
    for block_id in range(len(unet.up_blocks)):
        if not hasattr(unet.up_blocks[block_id], 'attentions'):
            continue
        for attention_id in range(len(unet.up_blocks[block_id].attentions)):
            for transformer_id in range(len(unet.up_blocks[block_id].attentions[attention_id].transformer_blocks)):
                unet.up_blocks[block_id].attentions[attention_id].transformer_blocks[transformer_id].attn2.to_k.requires_grad_(True)
                unet.up_blocks[block_id].attentions[attention_id].transformer_blocks[transformer_id].attn2.to_v.requires_grad_(True)

    #MID_BLOCK
    for attention_id in range(len(unet.mid_block.attentions)):
        for transformer_id in range(len(unet.up_blocks[block_id].attentions[attention_id].transformer_blocks)):
            unet.mid_block.attentions[attention_id].transformer_blocks[transformer_id].attn2.to_k.requires_grad_(True)
            unet.mid_block.attentions[attention_id].transformer_blocks[transformer_id].attn2.to_v.requires_grad_(True)

    return unet 

class Light2TokenBlock(torch.nn.Module):
    def __init__(self, out_dim=768, in_dim=27, num_token=1, hidden_layers=2, hidden_dim=256, *args, **kwargs):
        super().__init__()
        self.mlp = get_mlp(in_dim, hidden_dim,  hidden_layers, out_dim*num_token)
        self.mlp = initialize_weights_of_last_output(self.mlp, target = 0)

        # save parameter 
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.num_token = num_token


    def forward(self, x):
        """
        params:
            - x (torch.tensor): shape[bathch, in_dim]       
        """
        assert len(x.shape) == 2 
        assert x.shape[-1] == self.in_dim 
        batch_size = x.shape[0]
        y = self.mlp(x)
        y = y.reshape(batch_size, self.num_token, self.out_dim)
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

# Initialize weights and biases to ensure the output is always 0
def initialize_weights_of_last_output(model, target = 0):
    if not isinstance(model[-1], torch.nn.Linear):
        raise ValueError("The model should end with a linear layer to initialize the weights and biases.")
    model[-1].weight.data.fill_(0.0)  # Set weights to zero
    model[-1].bias.data.fill_(target)    # Set biases to zero
    return model