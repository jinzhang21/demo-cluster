import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = 5):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.policy_head = nn.Linear(64 * board_size * board_size, board_size * board_size)
        self.value_head = nn.Linear(64 * board_size * board_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value.squeeze(1)

    def predict(self, board_tensor: np.ndarray):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(board_tensor)
            policy, value = self.forward(x)
            policy = F.softmax(policy, dim=1)
        return policy.numpy(), value.numpy()
