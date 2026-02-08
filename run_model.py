import pandas as pd
import torch
import torch.nn
import torch.optim as optim

from models.residual_nn import MLP
from train_model import train_model

if __name__ == '__main__':
    df = pd.read_csv('./data/train_data.csv', index_col=False).iloc[:, 1:]
    X = df.drop(columns=['time_id', 'target', 'stock_id']) * 10
    stock = df.loc[:, ['stock_id']]
    y = df['target'] * 10

    X = X.to_numpy()
    stock = stock.to_numpy()
    y = y.to_numpy()

    X = torch.tensor(X, dtype=torch.float32)
    stock = torch.tensor(stock, dtype=torch.int).squeeze(-1)
    y = torch.tensor(y, dtype=torch.float32)

    from sklearn.model_selection import train_test_split

    X_train, X_test, stock_train, stock_test, y_train, y_test = train_test_split(X, stock, y, test_size=0.3,
                                                                                 shuffle=True, random_state=1)
    X_test, X_val, stock_test, stock_val, y_test, y_val = train_test_split(X_test, stock_test, y_test, test_size=0.5,
                                                                           shuffle=True, random_state=1)

    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(X_train, stock_train, y_train)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, pin_memory=True)

    val_dataset = TensorDataset(X_val, stock_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=2048, pin_memory=True)

    model = train_model(dataloader, val_dataloader, num_epochs=500, learning_rate=1e-3)

    model.eval()
    with torch.no_grad():
        model = model.to('cpu')
        test_pred = model(X_test, stock_test)
        test_loss = MLP.criterion(test_pred, y_test).item()
    print(test_loss)
    torch.save(model.state_dict(), f'first_model_loss_{test_loss:.4f}.pth')