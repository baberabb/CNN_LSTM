#LSTM model definition
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class RNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 rnn_hidden_size,
                 fc_hidden_size,
                 drop_out=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_out)
        # the first layer is embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM layer -> input:embedding -> output: Lstm hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        # RELU activation function
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(fc_hidden_size, 1)
        # sigmoid last layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        # pack sequences to save compute time
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden, cell) = self.rnn(out)
        # reshape LSTM output
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)  # self.dropout(out))
        out = self.sigmoid(out)
        return out

def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        # get the data
        text_batch, label_batch, lengths = text_batch.to(device), label_batch.to(device), lengths.to(device)
        # forward pass ->
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred.to(device), label_batch.float().to(device))
        # backward pass <-
        loss.backward()
        optimizer.step()
        total_acc += (
                (pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)

#evaluation loop
def evaluate(dataloader):
	model.eval()
	total_acc, total_loss = 0, 0
	with torch.no_grad():
		for text_batch, label_batch, lengths in dataloader:
			text_batch, label_batch, lengths = text_batch.to(device), label_batch.to(device), lengths.to(device)
			pred = model(text_batch, lengths)[:, 0]
			loss = loss_fn(pred, label_batch.float().to(device))
			total_acc += (
			(pred>=0.5).float() == label_batch).float().sum().item()
			total_loss += loss.item()*label_batch.size(0)
		return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

