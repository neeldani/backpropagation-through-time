import numpy as np

class RNNCell:
    def forward(self, x, h_prev, Wxh, Whh, bh, Why, by):
        self.x = x
        self.h_prev = h_prev
    
        self.z = np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh
        self.h = np.tanh(self.z)
        o = np.dot(Why, self.h) + by
        self.y_pred = self.softmax(o)

        return self.h, self.y_pred
    
    def cost(self, y_true):
        self.y_true = y_true
        self.loss = -np.sum(np.multiply(np.log(self.y_pred), self.y_true))
        return self.loss
    
    def backward(self, d_next_cell, Why, Whh):
        gradients = {}
        dL = self.y_pred - self.y_true

        # tier 1
        gradients["dby"] = dL
        gradients["dWhy"] = np.dot(dL, self.h.T)

        # tier 2
        dh = np.dot(Why.T, dL)
        dtanh = 1 - np.power(np.tanh(self.z), 2)
        dz = np.multiply(dtanh, dh + d_next_cell)
        
        gradients["dbh"] = dz
        gradients["dWxh"] = np.dot(dz, self.x.T)
        gradients["dWhh"] = np.dot(dz, self.h_prev.T)

        d_next_cell = np.dot(Whh.T, dz)
        
        # clip gradients
        gradients["dby"] = np.clip(gradients["dby"], -5, 5)
        gradients["dWhy"] = np.clip(gradients["dWhy"], -5, 5)
        
        gradients["dbh"] = np.clip(gradients["dbh"], -5, 5)
        gradients["dWxh"] = np.clip(gradients["dWxh"], -5, 5)
        gradients["dWhh"] = np.clip(gradients["dWhh"], -5, 5)

        return gradients, d_next_cell
    
    def softmax (self, z):
        return  np.exp(z)/np.sum(np.exp(z), axis = 0)

