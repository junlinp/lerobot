from torch.utils.data import Dataset, DataLoader
import tqdm
import numpy as np
import torch
import math
import os
import einops

def unpickle(file_path):
    import pickle
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

CIFAR_BASE="/home/junlinp/Downloads/cifar-10-python"

class CifarDataset(Dataset):
    def __init__(self, train:bool):

        train_path = [
        os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_1"),
           os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_2"),
           os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_3"),
           os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_4"),
           os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_5"),
        ]
        test_path = [
            os.path.join(CIFAR_BASE, "cifar-10-batches-py/test_batch"),
        ]

        self.train = train
        if self.train:
            data_dicts = [unpickle(item) for item in train_path]
            data_list =  [data[b'data'] for data in data_dicts]
            label_list = [data[b'labels'] for data in data_dicts]
            self.data = np.concatenate(data_list, axis = 0) / 255.0
            self.label = np.concatenate(label_list, axis = 0, dtype = np.int32)
        else:
            data_dicts = [unpickle(item) for item in test_path]
            data_list =  [data[b'data'] for data in data_dicts]
            label_list = [data[b'labels'] for data in data_dicts]
            self.data = np.concatenate(data_list, axis = 0) / 255.0
            self.label = np.concatenate(label_list, axis = 0, dtype=np.int32)
        print(f"data.shape : {self.data.shape}")
        print(f"label.shape : {self.label.shape}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index,:], self.label[index]

class FeedForward(torch.nn.Module):
    def __init__(self, model_dim:int = 512, hidden_dim:int = 2048):
        super(FeedForward, self).__init__()
        self.W1 = torch.nn.Linear(model_dim, hidden_dim)
        self.W2 = torch.nn.Linear(hidden_dim, model_dim)
        self.dropout = torch.nn.Dropout(0.1)
        # paper use a max operator
        # we use a ReLU replace
        self.activate = torch.nn.ReLU()

    def forward(self, x:torch.tensor):
        x = self.W1(x)
        x = self.activate(x)
        return self.W2(self.dropout(x))


class MultiHeadAttention(torch.nn.Module):

    def __init__(self,heads: int, model_dim: int, dropout:float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % heads == 0
        self.heads = heads
        self.k_dim = model_dim // heads
        self.query_weight = torch.nn.Linear(model_dim, model_dim)
        self.key_weight = torch.nn.Linear(model_dim, model_dim)
        self.value_weight = torch.nn.Linear(model_dim, model_dim)

        self.LN = torch.nn.LayerNorm(model_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.ff_layer = FeedForward(model_dim = model_dim)
        

    def forward(self, query, key, value, mask: torch.Tensor | None = None) -> torch.Tensor:
        assert query.size(-1) == self.heads * self.k_dim
        query = self.query_weight(query)
        key = self.key_weight(key)
        value = self.value_weight(value)
        query = einops.rearrange(query, "batch_size sequence_len (head_n head_dim) -> batch_size sequence_len head_n head_dim", head_n = self.heads, head_dim = self.k_dim)
        key = einops.rearrange(key, "batch_size sequence_len (head_n head_dim) -> batch_size sequence_len head_n head_dim", head_n = self.heads, head_dim = self.k_dim)
        value = einops.rearrange(value, "batch_size sequence_len (head_n head_dim) -> batch_size sequence_len head_n head_dim", head_n = self.heads, head_dim = self.k_dim)
        scores = einops.einsum(query, key, "batch_size sequence_length_1 head_n head_dim, batch_size sequence_length_2 head_n head_dim -> batch_size head_n sequence_length_1 sequence_length_2") / math.sqrt(self.k_dim)
        if mask is not None:
            scores.masked_fill(mask == 0, float('-inf'))

        logits = torch.nn.functional.softmax(scores, dim=-1)
        logits = self.dropout(logits)
        attention = einops.einsum(logits, value, "batch_size head_n seq_1 seq_2, batch_size seq_2 head_n head_dim -> batch_size seq_1 head_n head_dim")
        attention = einops.rearrange(attention, "batch_size seq_len head_n head_dim -> batch_size seq_len (head_n head_dim)")
        feed_forward_input = self.LN(attention)
        ff_output = self.ff_layer.forward(feed_forward_input)
        return feed_forward_input + ff_output



def cal_position_embedding(sequence_len:int, token_dim: int) -> torch.Tensor:
    assert token_dim % 2 == 0
    fraction = torch.linspace(0, token_dim - 1, steps = token_dim // 2)
    index = torch.linspace(0, sequence_len - 1, steps= sequence_len)

    fraction = einops.repeat(fraction, "token_dim_div_2 -> seq token_dim_div_2", seq = sequence_len, token_dim_div_2 = token_dim // 2) / token_dim
    index = einops.repeat(index, "seq -> seq token_dim_div_2", seq = sequence_len, token_dim_div_2 = token_dim // 2)
    value = index / (10000 ** fraction)
    embedding = torch.concat([torch.sin(value), torch.cos(value)], dim = 1)
    embedding.required_grad = False
    #embedding = torch.ones((sequence_len, token_dim))
    #for i in range(sequence_len):
        #for j in range(token_dim):
            #embedding[i][j] = np.sin(i / (10000 ** (j / token_dim))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / token_dim)))
    return embedding


class VisualTransform(torch.nn.Module):

    def __init__(self, patch_kernel_size: int, class_number: int):
        super(VisualTransform, self).__init__()
        self.token_dim = 512
        self.layer_num = 12
        self.patch_kernel_size = patch_kernel_size

        self.embedding_weight = torch.nn.Linear(3 * 16 * 16, self.token_dim, class_number)

        # conv implement
        #self.embedding_weight = torch.nn.Conv2d(3, self.token_dim, kernel_size=16, stride=16)

        self.attention_layer = torch.nn.ModuleList([
            MultiHeadAttention(8, self.token_dim) for i in range(self.layer_num)
        ])

        self.project = torch.nn.Linear(self.token_dim, class_number)
        self.activate = torch.nn.GELU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.class_token = torch.nn.Parameter(torch.randn(self.token_dim,))

        self.LN = torch.nn.LayerNorm(self.token_dim)

    def forward(self, x):
        # x should be (batch_size, C, H, W)
        #x = x.view(x.size(0), -1, 16 * 16)
        #x = self.embedding_weight(x).flatten(2).permute(0, 2, 1)
        x = einops.rearrange(x, "batch_size c (ph p1) (pw p2) -> batch_size (ph pw) (c p1 p2)", p1 = self.patch_kernel_size, p2 = self.patch_kernel_size)
        x = self.embedding_weight(x)
        batch_size = x.size(0)

        #class_token_expand = self.class_token.expand(x.size(0), 1, self.token_dim)
        class_token_expand = einops.repeat(self.class_token, "token_dim -> batch one token_dim", batch = batch_size, one = 1)

        x = torch.concat([class_token_expand, x], dim = 1)
        # TODO(junlinp):  position_embedding_table
        pos_embeding = cal_position_embedding(x.size(1), x.size(2)).to(x.device)
        x = x + pos_embeding

        # self-attension
        for layer in self.attention_layer:
            x = self.LN(x)
            x = layer.forward(x, x, x) + x

        projected_output = self.project(x[:, 0, :]) 
        return self.softmax(einops.rearrange(projected_output,"batch_size cls_dim token_dim -> batch_size (cls_dim token_dim)", cls_dim = 1))



def main():
    train_dataset = CifarDataset(train=True)
    test_dataset = CifarDataset(train=False)

    train_dataloader = DataLoader(train_dataset, batch_size = 512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle=False)

    train_features, train_label = next(iter(train_dataloader))
    

    device = torch.device("cuda")

    model = VisualTransform(patch_kernel_size=16, class_number=10)
    model = model.to(device)

    criterion = torch.nn.NLLLoss()

    optim = torch.optim.Adam(model.parameters(), lr = 1e-5)

    for epoch in range(128):
        avg_loss = 0.0
        train_iter = tqdm.tqdm(
            enumerate(train_dataloader),
            desc=f"EP {epoch}"
            )
        correct = 0
        for i, data in train_iter:
            x, y = data

            x = x.view(-1, 3, 32, 32).to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.int64)
            predict = model.forward(x)
            predict_class = torch.argmax(predict, dim = 1)
            loss = criterion(torch.log(predict), y)
            correct += torch.sum(predict_class == y)
            avg_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"EP {epoch},train avg_loss = {avg_loss / len(train_dataset)}, Correct {correct * 100.0 / len(train_dataset)} %")

        correct = 0
        for test_data in test_dataloader:
            x, y = test_data
            x = x.view(-1, 3, 32, 32).to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.int64)
            predict = model.forward(x)
            predict_class = torch.argmax(predict, dim = 1)
            correct += torch.sum(predict_class == y)

        print(f"EP {epoch},Valid Correct {correct * 100.0 / len(test_dataset)} %")




if __name__ == "__main__":
    main()