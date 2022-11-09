
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data.dataset import random_split


class CNN(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[200, 200, 200],
                 dropout=0.25):
        super(CNN, self).__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=self.embed_dim,
                                      padding_idx=0,
                                      max_norm=5.0)
        # Conv Layer
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # FCL
        self.fc = nn.Linear(np.sum(num_filters), 1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        # Get embeddings from input ids
        x_embed = self.embedding(input_ids).float()

        # Permute shape from (b, max_len, embed_dim) - > (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling.
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_poo
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b)
        logits = self.fc(self.dropout(x_fc))
        logits = self.sigmoid(logits)

        return logits


def train_CNN(config=None):
    with wandb.init(config=config):
        config = wandb.config
        vocab_size = len(vocab)
        torch.manual_seed(1)
        vocab_size = len(vocab)
        config.embed_dim = config.embed_dim
        config.rnn_hidden_size = 128
        config.fc_hidden_size = 128
        model = CNN(vocab_size, config.embed_dim, config.filter_sizes,
                    config.num_filters, config.dropout).to(device)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        def train(dataloader):
            model.train()
            total_acc, total_loss = 0, 0
            for text_batch, label_batch, lengths in dataloader:
                text_batch, label_batch, lengths = text_batch.to(device), label_batch.to(device), lengths.to(device)
                optimizer.zero_grad()
                pred = model(text_batch)[:, 0]
                loss = loss_fn(pred.to(device), label_batch.float().to(device))
                loss.backward()
                optimizer.step()
                total_acc += (
                        (pred >= 0.5).float() == label_batch).float().sum().item()
                total_loss += loss.item() * label_batch.size(0)
            return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)

        def evaluate(dataloader):
            model.eval()
            total_acc, total_loss = 0, 0
            with torch.no_grad():
                for text_batch, label_batch, lengths in dataloader:
                    text_batch, label_batch, lengths = text_batch.to(device), label_batch.to(device), lengths.to(device)
                    pred = model(text_batch)[:, 0]
                    loss = loss_fn(pred, label_batch.float().to(device))
                    total_acc += (
                            (pred >= 0.5).float() == label_batch).float().sum().item()
                    total_loss += loss.item() * label_batch.size(0)
                return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)

        num_epochs = config.epochs
        torch.manual_seed(1)
        for epoch in range(num_epochs):
            acc_train, loss_train = train(train_dl)
            acc_valid, loss_valid = evaluate(valid_dl)
            wandb.log({"val_loss": loss_valid,
                       "train_loss": loss_train,
                       "train_accuracy": acc_train,
                       "val_accuracy": acc_valid,
                       "epoch": epoch,
                       })
            print(f'Epoch {epoch} accuracy: {acc_train:.4f}'
                  f' val_accuracy: {acc_valid:.4f}')

