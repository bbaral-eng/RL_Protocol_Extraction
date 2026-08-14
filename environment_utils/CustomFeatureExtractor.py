# import necessary libraries
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.torch_layers import CombinedExtractor, NatureCNN, BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import TensorDict
import torch.nn as nn
import torch as th
import gymnasium 



" _________________________________________________________  Custom Feature Extractor: ________________________________________________________ "


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN inheriting from NatureCNN to reuse its structure (assertions, final linear layer, forward pass) but replacing the deep CNN layers 
    with a shallow, configurable one for sparse data. 
    """
    def __init__(
        self, 
        observation_space: gymnasium.spaces.Box,
        features_dim: int = 128,

        # adjustable paramters for CNN 
        kernel_size: int = 3, 
        stride: int = 1,
        padding: int = 0,
        out_channels: int = 32,
        normalized_image: bool = True,
    ) -> None: 
        
        super().__init__(observation_space, features_dim)


        # redefine CNN with custom architecture
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))




class CustomFeatureExtractor(CombinedExtractor):
    """
    Custom feature extractor that uses the CustomCNN for voltage history tensor and a simple MLP for other observations. 
    Additionally, probe position is normalized to [0, 1] range, by dividing by the max position value (grid size).
    Inherits from CombinedExtractor to handle dict observation spaces.
    """
    def __init__(
        self, 
        observation_space: gymnasium.spaces.Dict,
        features_dim: int = 128,

        # custom CNN parameters as args 
        cnn_kwargs: dict = None,
        mlp_net_arch: list = None, 
        normalized_image: bool = True,
    ) -> None: 
        
        
        super().__init__(observation_space, features_dim)

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {
            'out_channels': 32, 
            'kernel_size': 3,
            'stride': 1,       
            'padding': 0,
            'features_dim': 128,
        }
        
        mlp_net_arch = mlp_net_arch if mlp_net_arch is not None else [64, 32] # default MLP for non-tensor data
        self.grid_size_for_normalization = 11 - 1 #  max position value is 10 (0-indexed)

        self.cnn_extractor = CustomCNN(
            observation_space.spaces['voltage history'],
            normalized_image = normalized_image,
            **{k: v for k, v in cnn_kwargs.items()},
        )

        self.cnn_output_dim = self.cnn_extractor._features_dim

        # MLP for other observations
        self.other_obs_dim = 4

        self.mlp_extractor = create_mlp(
            input_dim=self.other_obs_dim,
            output_dim = 0,
            net_arch=mlp_net_arch,
            activation_fn=nn.ReLU,
        )

        self.mlp_extractor_sequential = nn.Sequential(*self.mlp_extractor)
        self.mlp_output_dim = mlp_net_arch[-1]
        self._features_dim = self.cnn_output_dim + self.mlp_output_dim

    def forward(self, observations: TensorDict) -> th.Tensor:
    
        """forward pass that extracts features from both voltage history (via CNN) and other observations (via MLP) and concatenates them"""
            
        cnn_features = self.cnn_extractor((observations["voltage history"]))

        # normalize probe position to [0, 1] range by dividing by max position value (grid size)
        probe_position = observations["probe position"].float() / self.grid_size_for_normalization

        other_obs = [
            observations['voltage'],
            probe_position, 
            observations['time step'],
        ]

        other_features_tensor = th.cat(other_obs, dim=1)

        mlp_features = self.mlp_extractor_sequential(other_features_tensor)

        return th.cat((cnn_features, mlp_features), dim=1)

        



