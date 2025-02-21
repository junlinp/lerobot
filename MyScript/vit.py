from torch.utils.data import Dataset, DataLoader
import tqdm
import numpy as np
import torch
import math

def unpickle(file_path):
    import pickle
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CifarDataset(Dataset):
    def __init__(self, train:bool):

        train_path = [
            "./datasets/cifar-10-batches-py/data_batch_1",
            "./datasets/cifar-10-batches-py/data_batch_2",
            "./datasets/cifar-10-batches-py/data_batch_3",
            "./datasets/cifar-10-batches-py/data_batch_4",
            "./datasets/cifar-10-batches-py/data_batch_5",
        ]
        test_path = [
            "./datasets/cifar-10-batches-py/test_batch",
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
        

    def forward(self, query, key, value):
        assert query.size(-1) == self.heads * self.k_dim
        # query  (batch_size, sequence_length, token_dim) -> (batch_size, sequence_length, head_n * k_dim)
        # key -> (batch_size, sequence_length, token_dim) -> (batch_size, sequence_length, head_n * k_dim)
        # key -> (batch_size, sequence_length, token_dim) -> (batch_size, sequence_length, head_n * k_dim)
        query = self.query_weight(query)
        key = self.key_weight(key)
        value = self.value_weight(value)

        # (batch_size, sequence_length, head_n * k_dim) -> (batch_size, sequence_length, head_n, k_dim)
        query = query.view(query.size(0), -1, self.heads, self.k_dim)
        key = key.view(key.size(0), -1, self.heads, self.k_dim)
        value = value.view(value.size(0), -1, self.heads, self.k_dim)

        # -> (batch_size, head_n, sequence_length, k_dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # key transpose
        # scores (batch_size, head_n, sequence_length, sequence_length)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        logits = torch.nn.functional.softmax(scores, dim=-1)
        logits = self.dropout(logits)

        #(batch_size, head_n, sequence_length, k_dim)
        attention = torch.matmul(logits, value)

        # concate head_n 
        attention = attention.permute(0, 2, 1, 3).contiguous().view(attention.size(0), -1, self.heads * self.k_dim)

        #
        #print(f"attention.shape:{attention.shape}, query.shape: {query.shape}")
        # need to res
        feed_forward_input = self.LN(attention)

        ff_output = self.ff_layer.forward(feed_forward_input)

        return feed_forward_input + ff_output





class VisualTransform(torch.nn.Module):

    def __init__(self, class_number: int):
        super(VisualTransform, self).__init__()

        self.token_dim = 512
        self.layer_num = 12

        #self.embedding_weight = torch.nn.Linear(16 * 16, self.token_dim, class_number)
        self.embedding_weight = torch.nn.Conv2d(3, self.token_dim, kernel_size=16, stride=16)

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
        x = self.embedding_weight(x).flatten(2).permute(0, 2, 1)

        class_token_expand = self.class_token.expand(x.size(0), 1, self.token_dim)

        x = torch.concat([class_token_expand, x], dim = 1)

        # self-attension
        for layer in self.attention_layer:
            x = self.LN(x)
            x = layer.forward(x, x, x) + x

        # output (batch_size, sequence_length, token_dim)
        
        return self.softmax(self.project(x[:, 0, :]).view(x.size(0), -1))



def main():
    train_dataset = CifarDataset(train=True)
    test_dataset = CifarDataset(train=False)

    train_dataloader = DataLoader(train_dataset, batch_size = 512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle=False)

    train_features, train_label = next(iter(train_dataloader))
    

    device = torch.device("cuda")

    model = VisualTransform(class_number=10)
    model = model.to(device)

    criterion = torch.nn.NLLLoss()

    optim = torch.optim.Adam(model.parameters(), lr = 1e-5)

    for epoch in range(128):
        avg_loss = 0.0
        train_iter = tqdm.tqdm(
            enumerate(train_dataloader),
            desc=f"EP_{epoch}"
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

        print(f"EP{epoch}, avg_loss = {avg_loss / len(train_dataset)}, Correct {correct * 100.0 / len(train_dataset)} %")

        correct = 0
        for test_data in test_dataloader:
            x, y = test_data
            x = x.view(-1, 3, 32, 32).to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.int64)
            predict = model.forward(x)
            predict_class = torch.argmax(predict, dim = 1)
            correct += torch.sum(predict_class == y)

        print(f"Correct {correct * 100.0 / len(test_dataset)} %")




if __name__ == "__main__":
    main()