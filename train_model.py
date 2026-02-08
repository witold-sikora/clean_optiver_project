import pandas as pd
import torch
import torch.nn
import torch.optim as optim

from models.residual_nn import  MLP

num_epochs = 100
batch_size = 2048
learning_rate = 1e-4
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def train_model(
        train_loader,
        val_loader,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.1,
        emb_dim=16,
        emb_num=127,
        num_epochs=100,
        learning_rate=1e-4,
        device=device):
    ## defining parameters
    in_dim = train_loader.dataset[0][0].shape[-1]
    model = MLP(in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=dropout,
                emb_dim=emb_dim,
                emb_num=emb_num).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-5)
    criterion = MLP.criterion

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=3,
        threshold=1e-5,
        eps = 0
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, stock_id, y in train_loader:
            x, stock_id, y = x.to(device), stock_id.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, stock_id)
            loss = criterion(output, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        for x, stock_id, y in val_loader:
            x, stock_id, y = x.to(device), stock_id.to(device), y.to(device)
            with torch.no_grad():
                output = model(x, stock_id)
                loss = criterion(output, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch: {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss} | LR: {current_lr}')

        if current_lr < 2e-8:
            print(f'LR = {current_lr} is too small! Early stopping the training.')
            break

    return model