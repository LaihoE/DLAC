import numpy as np
import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, _ = self.gru(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out


"""
Model predicts the probability of cheating for every shot that hits an enemy
"""
if __name__ == "__main__":
    # Check for GPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using: {device}")

    # Shape expected is (n,256,24). Example has 20 shots so the shape is (20,256,24)
    X = np.load("example.npy")
    X = torch.tensor(X).to(device).float()

    # Deep learning model. Remove map_location if GPU enabled PyTorch. Enabling GPU speeds up predictions, but may
    # not be needed if predicting small amounts of games
    model = torch.load("AntiCheat3.pt",map_location=torch.device('cpu'))
    prd = model.forward(X)
    probs = torch.softmax(prd,1)

    # Each kill in numpy array
    for shot in range(X.shape[0]):
        probability = probs[shot][1].item()

        # The probabilities are way too confident. For example use 95 % as threshold for a cheating shot.
        # You can come up with any rule you want, for example if average is over X% or if top 5 predictions are over
        # X% or even create a ML model on top of these

        if probability > 0.95:
            print("Shot number:", shot, "Cheating:", round(probability, 2)*100, "%")