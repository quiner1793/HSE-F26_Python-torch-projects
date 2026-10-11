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


class PriceRNN(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        output, _ = self.rnn(x)
        last_output = output[:, -1, :]
        return self.head(last_output).squeeze(1)


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


class PriceGRU(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        output, _ = self.gru(x)
        last_output = output[:, -1, :]
        return self.head(last_output).squeeze(1)


def build_model(model_name: str, window: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "mlp":
        return PriceMLP(window=window)
    if model_name == "rnn":
        return PriceRNN()
    if model_name == "lstm":
        return PriceLSTM()
    if model_name == "gru":
        return PriceGRU()

    raise ValueError(f"Unknown model: {model_name}")
