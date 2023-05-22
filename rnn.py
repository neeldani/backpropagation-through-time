import numpy as np
from rnn_cell import RNNCell 

class RNN:
    def __init__(self, n_in, n_hidden, n_out, n_timestamps, weights=None):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_timestamps = n_timestamps
        
        if weights is None:
            self.Wxh = np.random.randn(self.n_hidden, self.n_in) * 0.01
            self.Whh = np.random.randn(self.n_hidden, self.n_hidden) * 0.01
            self.bh = np.random.randn(self.n_hidden, 1) * 0.01

            self.Why = np.random.randn(self.n_out, self.n_hidden) * 0.01
            self.by = np.random.randn(self.n_out, 1) * 0.01
        
        else:
            self.Wxh = weights["Wxh"]
            self.Whh = weights["Whh"]
            self.bh = weights["bh"]
            
            self.Why = weights["Why"]
            self.by = weights["by"]

        self.model = [ RNNCell() for _ in range(self.n_timestamps)]

    def train(self, x, y, alpha):
        cost = self.forward(x, y)
        dWxh, dWhh, dbh, dWhy, dby = self.bptt()
        
        self.Wxh = self.Wxh - alpha * dWxh
        self.Whh = self.Whh - alpha * dWhh
        self.bh = self.bh - alpha * dbh

        self.Why = self.Why - alpha * dWhy
        self.by = self.by - alpha * dby
        
        return cost
    
    def forward(self, x, y):
        # forward
        h = np.zeros((self.n_hidden, 1))
        total_cost = 0
        
        for t in range(self.n_timestamps):
            rnn_cell = self.model[t]
            h, _ = rnn_cell.forward(x[:, [t]], h, self.Wxh, self.Whh, self.bh, self.Why, self.by)
            total_cost += rnn_cell.cost(y[:, [t]])
            
        return total_cost/self.n_timestamps
    
    def predict(self, x, h):
        y_preds = []
        for t in range(self.n_timestamps):
            rnn_cell = self.model[t]
            h, y_pred = rnn_cell.forward(x, h, self.Wxh, self.Whh, self.bh, self.Why, self.by)
            y_preds.append(y_pred)
            
        return y_preds

    def bptt(self):
        # bptt
        T = self.n_timestamps
        d_next_cell = np.zeros((self.n_hidden, 1))
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        for t in range(T-1, -1, -1):
            gradients, d_next_cell = self.model[t].backward(d_next_cell, self.Why, self.Whh)

            dWxh += gradients["dWxh"]
            dWhh += gradients["dWhh"]
            dbh += gradients["dbh"]
            
            dWhy += gradients["dWhy"]
            dby += gradients["dby"]
        
        return dWxh/T, dWhh/T, dbh/T, dWhy/T, dby/T