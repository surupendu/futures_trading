from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from gym import spaces
import torch

class CNNStateLayer(BaseFeaturesExtractor):
    '''
        CNN layer to extract features from state vector to form feature vector
    '''
    def __init__(self, observation_space: spaces.Dict, feature_dim):
        '''
            Initialize the CNN layer
        '''
        super(CNNStateLayer, self).__init__(observation_space, feature_dim)
        num_input_channels = observation_space["market_value"].shape[1]
        self.conv_1 = nn.Conv1d(in_channels=num_input_channels, out_channels=20, kernel_size=3)
        self.linear_1 = nn.Linear(21, feature_dim)
    
    def forward(self, observation):
        '''
            Extract feature vector from the state
        '''
        market_values = observation["market_value"]
        market_values = market_values.transpose(1, 2)
        market_values = self.conv_1(market_values)
        market_values = market_values.transpose(1, 2)
        market_value = torch.mean(market_values, dim=1)

        action = observation["action"]
        obs_vector = torch.cat((market_value, action), dim=1)
        obs_vector = self.linear_1(obs_vector)
        return obs_vector
