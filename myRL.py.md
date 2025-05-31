# Documentation for `myRL.py`

## Overview

`myRL.py` is a Python script that implements components for Reinforcement Learning (RL) tasks, likely related to the TBFIT package's goal of parameter fitting or material design. It leverages the PyTorch library to define and train RL agents. The script includes implementations for:

- A custom PyTorch Dataset (`mytb_dataset`) for loading tight-binding band structure data.
- A Deep Q-Network (DQN) agent (`mydqn`).
- A Deep Deterministic Policy Gradient (DDPG) agent (`myddpg`).
- Utility functions for CNN output size calculation, model parameter counting, and device selection.

This module seems intended to explore or apply RL techniques to problems involving band structure data, potentially for optimizing tight-binding parameters or discovering materials with desired electronic properties.

## Key Components

### 1. `mytb_dataset(Dataset)` Class
- **Purpose:** A custom PyTorch `Dataset` for loading and serving tight-binding (TB) band data and corresponding target (e.g., DFT) band data.
- **Initialization (`__init__`)**:
    - Takes `filename` (path to a `.pt` file saved by `torch.save`) and `device`.
    - Loads data, expecting a dictionary with keys 'TBA' (model-calculated bands) and 'DFT' (target bands).
- **Methods:**
    - `__getitem__(index)`: Returns a sample (a 'TBA' band and the first 'DFT' band as target) at the given index.
    - `__len__()`: Returns the total number of samples.

### 2. `mydqn` Class
- **Purpose:** Implements a Deep Q-Network (DQN) agent.
- **Initialization (`__init__`)**:
    - `buffer_limit`: Maximum size of the replay buffer.
    - `batch_size`: Batch size for training.
    - `N_BANDS`, `N_KPOINTS`: Dimensions of the band structure data (state representation).
    - `N_W`: Likely related to the number of actions or output dimension scaling.
    - `gamma`: Discount factor for future rewards.
    - Initializes a `_QNet` (the neural network for Q-value approximation) and a `_ReplayBuffer`.
- **Inner Classes:**
    - `_ReplayBuffer`:
        - Stores experiences (state, action, reward, next_state, done_mask) in a `collections.deque`.
        - `put(transition)`: Adds a transition to the buffer.
        - `sample(n)`: Samples a mini-batch of transitions from the buffer.
        - `size()`: Returns the current size of the buffer.
    - `_QNet(nn.Module)`:
        - A simple fully connected neural network (3 linear layers with ReLU activations) to approximate Q-values.
        - `forward(x)`: Defines the forward pass.
        - `sample_action(obs, epsilon)`: Implements an epsilon-greedy policy to select an action.
- **Methods:**
    - `train(q, q_target, memory, optimizer)`: Performs a single training step for the Q-network. Samples from replay buffer, calculates target Q-values, computes loss (Smooth L1 loss), and updates the network.
    - `Reporter(...)`: A utility function for printing messages, possibly for logging training progress.

### 3. `myddpg` Class
- **Purpose:** Implements a Deep Deterministic Policy Gradient (DDPG) agent, suitable for continuous action spaces.
- **Initialization (`__init__`)**:
    - Similar parameters to `mydqn`, plus `tau` (for soft target network updates).
    - Initializes `_MuNet` (actor network), `_QNet` (critic network), and `_ReplayBuffer`.
- **Inner Classes:**
    - `_ReplayBuffer`: Identical in functionality to the one in `mydqn`.
    - `_MuNet(nn.Module)` (Actor Network):
        - A neural network that maps states to actions.
        - Uses `torch.sigmoid` on the output, scaled by `N_BANDS*N_KPOINTS*N_W`, suggesting the action space might be related to indices or scaled continuous values.
    - `_QNet(nn.Module)` (Critic Network):
        - Approximates the action-value function (Q-value).
        - Takes both state (`x`) and action (`a`) as input.
        - Concatenates processed state and action features before further layers.
    - `OrnsteinUhlenbeckNoise`:
        - Implements Ornstein-Uhlenbeck noise, often used for exploration in DDPG by adding noise to actions.
- **Methods:**
    - `train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)`: Performs a training step for both actor and critic networks.
        - Updates the critic (`q`) similarly to DQN, using the actor_target (`mu_target`) for next state actions.
        - Updates the actor (`mu`) by maximizing the critic's output.
    - `soft_update(net, net_target)`: Performs soft updates for target networks (a common practice in DDPG to stabilize training).
    - `Reporter(...)`: Same utility as in `mydqn`.

### 4. Utility Functions
- `nout_cnn(in_features, network, pooling=None)`:
    - Calculates the output dimension of a 1D or 2D convolutional layer given input features and layer parameters (padding, kernel_size, stride, pooling).
- `count_parameters(model)`:
    - Prints the model architecture and the total number of trainable parameters.
- `get_device()`:
    - Returns a PyTorch device object (`cuda` if available, otherwise `cpu`).

## Important Variables/Constants

- **Within RL classes (`mydqn`, `myddpg`):**
    - `self.device`: Stores the PyTorch device (`cuda` or `cpu`).
    - `self.QNet`, `self.MuNet`: Instances of the neural networks.
    - `self.ReplayBuffer`: Instance of the replay buffer.
    - Hyperparameters like `gamma`, `tau`, `batch_size`, `buffer_limit`.
    - `N_BANDS`, `N_KPOINTS`, `N_W`: Define the assumed structure/size of states and potentially actions.

## Usage Examples

**1. Using `mytb_dataset`:**
```python
# Assuming 'tb_data.pt' is a file created by torch.save({'TBA': bands_tensor, 'DFT': dft_bands_tensor})
dataset = mytb_dataset(filename='tb_data.pt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_bands, batch_target in dataloader:
    # batch_bands and batch_target are now ready for training a model
    pass
```

**2. Setting up a DQN agent:**
```python
# Define state and action space parameters
n_bands = 10
n_kpoints = 50
n_actions_components = 3 # Example value for N_W

dqn_agent = mydqn(N_BANDS=n_bands, N_KPOINTS=n_kpoints, N_W=n_actions_components)
q_optimizer = optim.Adam(dqn_agent.QNet.parameters(), lr=0.001)

# Example training loop (highly simplified)
# for episode in range(num_episodes):
#     current_state = env.reset() # Assuming an environment 'env'
#     done = False
#     while not done:
#         action = dqn_agent.QNet.sample_action(torch.tensor(current_state, dtype=torch.float), epsilon)
#         next_state, reward, done, _ = env.step(action)
#         dqn_agent.ReplayBuffer.put((current_state, action, reward, next_state, done))
#
#         if dqn_agent.ReplayBuffer.size() > dqn_agent.batch_size:
#             dqn_agent.train(dqn_agent.QNet, q_target_network, dqn_agent.ReplayBuffer, q_optimizer)
#         current_state = next_state
```
*(Note: A target Q-network (`q_target_network`) and an environment (`env`) would need to be defined for a complete DQN setup.)*

**3. Setting up a DDPG agent:**
```python
ddpg_agent = myddpg(N_BANDS=n_bands, N_KPOINTS=n_kpoints, N_W=n_actions_components)
mu_optimizer = optim.Adam(ddpg_agent.MuNet.parameters(), lr=0.0005)
q_optimizer = optim.Adam(ddpg_agent.QNet.parameters(), lr=0.001)
ounoise = ddpg_agent.OrnsteinUhlenbeckNoise(mu=np.zeros(1)) # Assuming action is scalar for noise

# (Similar training loop structure to DQN, but with actor-critic updates and noise for exploration)
```

## Dependencies and Interactions

- **External Libraries:**
    - `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`: Core PyTorch components for neural networks and optimization.
    - `numpy`: For numerical operations, especially with noise generation in DDPG.
    - `collections.deque`: For the replay buffer.
    - `random`: For random sampling and epsilon-greedy exploration.
    - `sys`: For `sys.stdout.flush()` in `Reporter`.
- **Data:**
    - Expects band structure data in a specific format (`.pt` files with 'TBA' and 'DFT' keys) for `mytb_dataset`.
- **Potential Interactions:**
    - This module could be used in conjunction with `tbfitpy_serial.py` or `tbfitpy_mpi.py`. For instance, the `pytbfit` class could serve as the environment for the RL agent, where actions modify tight-binding parameters and the reward is based on the quality of the fit to DFT bands. The `generate_TBdata` method in `pytbfit` might be used to create datasets for `mytb_dataset`.

This script provides a toolkit for applying advanced reinforcement learning algorithms to problems likely involving the optimization or exploration of tight-binding models and their parameters.
