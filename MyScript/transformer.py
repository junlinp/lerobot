import torch
import einops
import math
class FeedForward(torch.nn.Module):
    def __init__(self, model_dim:int = 512, hidden_dim:int = 2048, dropout_rate:float = 0.1):
        super(FeedForward, self).__init__()
        self.W1 = torch.nn.Linear(model_dim, hidden_dim)
        self.W2 = torch.nn.Linear(hidden_dim, model_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        # paper use a max operator
        # we use a ReLU replace
        self.activate = torch.nn.ReLU()

    def forward(self, x:torch.tensor):
        x = self.W1(x)
        x = self.activate(x)
        return self.W2(self.dropout(x))



class MultiHeadAttention(torch.nn.Module):
    def __init__(self, feed_forward_dim:int,num_heads: int, model_dim: int, dropout:float = 0.1, attention_dropout_rate: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        assert model_dim % num_heads == 0
        self.heads = num_heads
        self.k_dim = model_dim // num_heads
        self.query_weight = torch.nn.Linear(model_dim, model_dim)
        self.key_weight = torch.nn.Linear(model_dim, model_dim)
        self.value_weight = torch.nn.Linear(model_dim, model_dim)

        self.LN = torch.nn.LayerNorm(model_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.ff_layer = FeedForward(model_dim = model_dim, hidden_dim = feed_forward_dim, dropout_rate=dropout)
        self.layer_norm = torch.nn.LayerNorm([model_dim])
        self.attention_dropout = torch.nn.Dropout(attention_dropout_rate)
        

    def forward(self, x:torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        input = x
        x = self.layer_norm(x)
        #assert query.size(-1) == self.heads * self.k_dim
        query = self.query_weight(x)
        key = self.key_weight(x)
        value = self.value_weight(x)

        query = einops.rearrange(query, "batch_size sequence_len (head_n head_dim) -> batch_size sequence_len head_n head_dim", head_n = self.heads, head_dim = self.k_dim)
        key = einops.rearrange(key, "batch_size sequence_len (head_n head_dim) -> batch_size sequence_len head_n head_dim", head_n = self.heads, head_dim = self.k_dim)
        value = einops.rearrange(value, "batch_size sequence_len (head_n head_dim) -> batch_size sequence_len head_n head_dim", head_n = self.heads, head_dim = self.k_dim)
        scores = einops.einsum(query, key, "batch_size sequence_length_1 head_n head_dim, batch_size sequence_length_2 head_n head_dim -> batch_size head_n sequence_length_1 sequence_length_2") / math.sqrt(self.k_dim)
        if attention_mask is not None:
            scores.masked_fill(attention_mask == 0, float('-inf'))
        logits = torch.nn.functional.softmax(scores, dim=-1)
        logits = self.attention_dropout(logits)
        attention = einops.einsum(logits, value, "batch_size head_n seq_1 seq_2, batch_size seq_2 head_n head_dim -> batch_size seq_1 head_n head_dim")
        attention = einops.rearrange(attention, "batch_size seq_len head_n head_dim -> batch_size seq_len (head_n head_dim)")
        attention = self.dropout(attention)
        y = attention + input
        feed_forward_input = self.LN(y)
        ff_output = self.ff_layer.forward(feed_forward_input)
        return feed_forward_input + ff_output



class Transformer(torch.nn.Module):
    def __init__(self, input_dim:int,num_layers:int,feed_forward_dim:int, num_attention_heads:int, dropout_rate:float = 0.1, attention_dropout_rate: float = 0.1, add_position_embedding:bool = False):
        super(Transformer,self).__init__()

        self.add_position_embedding = add_position_embedding
        self.num_layers = num_layers

        self.attention_layers = torch.nn.ModuleList(
            [MultiHeadAttention(feed_forward_dim=feed_forward_dim, num_heads=num_attention_heads, model_dim=input_dim, dropout=dropout_rate, attention_dropout_rate=attention_dropout_rate) for i in range(num_layers)]
        )
        self.layer_norm = torch.nn.LayerNorm([input_dim])


    def forward(self, x:torch.Tensor, attention_mask: torch.Tensor|None = None) -> torch.Tensor:
        #assert x.size() == 3

        if self.add_position_embedding:
            # TODO (position embedding)
            pass
        
        for lyr in self.attention_layers:
            x = lyr.forward(x, attention_mask)
        x = self.layer_norm(x)
        return x

        


    