import torch
import torch.nn as nn

from lib.models.sm_lstm.encoder import RegLSTMEncoder


class SiameseRegLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = RegLSTMEncoder(config)
        self.dropout = nn.Dropout(config.dropout)
        if config.is_bidirectional:
            self.fc1 = nn.Linear(2 * 2 * config.hidden_dim, config.target_class)
        else:
            self.fc1 = nn.Linear(2 * config.hidden_dim, config.target_class)

        if config.beta_ema>0:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, query, text):
        query = self.encoder(query)  # (batch, channel_output * num_conv)
        text = self.encoder(text)  # (batch, channel_output * num_conv)
        x = torch.cat([query, text], 1)   # (batch, 2 * channel_output * num_conv)
        x = self.dropout(x)
        return self.fc1(x)  # (batch, target_size)

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1-self.beta_ema)*p.data)
    
    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p/(1-self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p,avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params
