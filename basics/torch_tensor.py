import torch

device = torch.device("cuda:0")

epoch = 10
n_x = 25  # Features in each training example.
m = 1000  # Number of Training Examples.

# inputs : Stacked column-wise
X = torch.rand([m, n_x, 1], device=device)
Y = torch.ones([1, m], device=device)
print(X.shape)

# Weight Matrix : Single Neuron Weight Matrix
W = torch.rand([n_x, 1], device=device)
print(W.shape)

# Bias Term : Single Neuron.
b = torch.rand([1, m], device=device)
print(b.shape)

for i in range(epoch):
    print()
    # Z = W_t * X + b
    Z = torch.matmul(torch.transpose(W, 0, 1), X) + b
    print("Z", Z.shape)

    # Sigmoid for binary classification
    A = torch.sigmoid(Z)
    del_loss = A - Y  # From Gradient Calculation.

    print("del_loss", del_loss.shape)
    print("X", X.shape)
    print("Y", Y.shape)

    del_weights = (1 / m) * torch.matmul(X, del_loss).sum()
    del_b = (1 / m) * del_loss.sum()

    print("del_weights", del_weights.shape)
    W = W - 0.0001 * del_weights
    b = b - 0.0001 * del_b

    print("Weights", W.shape)
    print("Bias", b.shape)

    print(W)
