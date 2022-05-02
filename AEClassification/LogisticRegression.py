import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.input_dim  = input_dim
        self.output_dim  = output_dim
        self.activation = torch.nn.Sigmoid() if self.output_dim == 1 else torch.nn.Softmax(dim=-1)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.activation(self.linear(x))
        return outputs
