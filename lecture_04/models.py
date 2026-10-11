from torch import Tensor, nn


class PriceMLP(nn.Module):
    def __init__(self, window: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(1)


class PriceLSTM(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.head(last_output).squeeze(1)
