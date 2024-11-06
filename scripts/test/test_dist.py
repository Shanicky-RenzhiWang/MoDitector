import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        action_probs = F.softmax(x, dim=-1)
        return action_probs

    def _get_action(self, features):
        return self.forward(features)


# Example usage
input_dim = 4  # Example input feature dimension
output_dim = 3  # Example number of actions
policy = PolicyNetwork(input_dim, output_dim)

features = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Get action probabilities
actions_probs = policy._get_action(features)

# Create a categorical distribution over actions
dist = Categorical(actions_probs)

# Assume an action was taken (e.g., action index 1)
actions = torch.tensor([0.1, 0.2, 0.3])

# Compute log probability of the taken action
log_prob = dist.log_prob(actions)

# Compute the entropy of the action distribution
dist_entropy = dist.entropy()

print(f"Action probabilities: {actions_probs}")
print(f"Log probability of action: {log_prob}")
print(f"Entropy of action distribution: {dist_entropy}")