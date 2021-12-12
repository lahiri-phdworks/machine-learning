import torch
import torch.nn as nn
device = torch.device('cuda:0')

# Input Size, hidden_size
RNNcell = nn.RNN(input_size=125, hidden_size=60, batch_first=True).cuda()

# Float type input, make it torch.Long if Long (Int)
# inputs = torch.tensor([[[1., 0., 1., 0., 1.]]], dtype=torch.float32)

# for multiple inputs (batch_size > 1) && sequence length >= 1
# Sequence Length => # RNN Cells in a layer. 
# batch_size, sequence_length (number of RNN Cells), Input Size (vector length)
inputs = torch.rand([500, 50, 125], dtype=torch.float32, device=device)
print(inputs.size())
# (D * num_layers (2:Bi-Directional), batch, hidden_size:out)
hidden_init = torch.rand([1, 500, 60], dtype=torch.float32, device=device)
print(hidden_init.size())
## Output from single RNN Cell (One cell, single layer, nothing fancy)
#for i in range(10):
#    inputs = torch.rand([1, 1, 5], dtype=torch.float32)
#    out, hidden_out = cell(inputs, hidden_init)
#    print(f"Out : {i + 1}th, {out.data}")
#    print(f"Hidden : {i + 1}th, {hidden_out.data}")

# Output from the RNN Layer.
out, hidden_out = RNNcell(inputs, hidden_init)
print(f"Out : {out.detach().data}")
print(f"Hidden : {hidden_out.detach().data}")
