import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

# for continuous action space, the action_std is the standard deviation of the action distribution
# for discrete action space, the action_std is not used, but it is still required to
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class ActorCritic(nn.Module):
    """
    Simple MLP application for Actor-Critic Policy
    This class implements a simple MLP for both actor and critic.
    """
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic for reducing the training variance
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, deterministic=False):
        """
        Select action from the policy for a given state.
        
        Args:
            state: Current state observation
            deterministic: If True, return mean/mode action; if False, sample from distribution
        
        Returns:
            action: Selected action (sampled or deterministic)
            action_logprob: Log probability of the action
            state_val: Value estimate for the state
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            if deterministic:
                # Deterministic: return mean action directly
                action = action_mean
                # For deterministic actions, log_prob is not meaningful, set to 0
                action_logprob = torch.zeros_like(action_mean[:, 0])
            else:
                # Stochastic: sample from Gaussian distribution
                cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
                dist = MultivariateNormal(action_mean, cov_mat)
                action = dist.sample()
                action_logprob = dist.log_prob(action)
        else:
            action_probs = self.actor(state)
            
            if deterministic:
                # Deterministic: return action with highest probability
                action = action_probs.argmax(dim=-1)
                # For deterministic actions, compute log_prob of the selected action
                action_logprob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
            else:
                # Stochastic: sample from categorical distribution
                dist = Categorical(action_probs)
                action = dist.sample()
                action_logprob = dist.log_prob(action)

        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy