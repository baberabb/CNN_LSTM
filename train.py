if __name__ == '__main__':
    # Initialize the model
    vocab_size = len(vocab)
    embed_dim = 20
    rnn_hidden_size = 128
    fc_hidden_size = 128
    torch.manual_seed(1)
    model = CNN(vocab_size, embed_dim).to(device)

    # Training loop
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    torch.manual_seed(1)
    for epoch in range(num_epochs):
        acc_train, loss_train = train(train_dl)
        acc_valid, loss_valid = evaluate(valid_dl)
        print(f'Epoch {epoch} accuracy: {acc_train:.4f}'
              f' val_accuracy: {acc_valid:.4f}')