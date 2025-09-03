import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "mps"

class RACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, policy_id=-1):
        super(RACPolicy, self).__init__()
        self.policy_id = policy_id  # Parent policy when id = -1, Child policy id >= 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_action = True

        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_dim)

    def forward(self, x):
        with torch.no_grad():  # Will not call Tensor.backward()
            # print(x.shape)
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(0)  # Todo: check x dim as its condition
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)  # all the dimensions of input of size 1 removed.
                x = torch.argmax(x)
            else:
                x = torch.relu(x.squeeze())

            x = x.detach().cpu().numpy()

            return x.item(0)
        
    def norm_init(self, std=1.0):
        for param in self.parameters():
            shape = param.shape
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            param.data = torch.from_numpy(out)

    def restore_params(self, init_path):
        # print(init_path)
        self.load_state_dict(torch.load(init_path))


    def set_policy_id(self, policy_id):
        self.policy_id = policy_id

    def reset(self):
        pass

    def get_param_list(self):
        param_lst = []
        for param in self.parameters():
            param_lst.append(param.data.numpy())
        return param_lst

    def set_param_list(self, param_lst: list):
        lst_idx = 0
        for param in self.parameters():
            param.data = torch.tensor(param_lst[lst_idx]).float()
            lst_idx += 1
