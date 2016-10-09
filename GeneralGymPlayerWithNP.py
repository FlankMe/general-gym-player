"""
General OpenAI gym player

Simple feed forward neural net that solves OpenAI gym environments 
(https://gym.openai.com) via Q-learning 

Download the code, assign to game_name the name of environment you wish to 
run, and let the script learn how to solve it.  
Note the code only works for environments with discrete action space and 
continuous observation space.

https://github.com/FlankMe/general-gym-player
@author: Riccardo Rossi
"""

# Choice of the game and definition of the goal
game_name = 'CartPole-v0'
MAX_EPISODES = 1500
CONSECUTIVE_EPISODES = 100   # Number of trials' rewards to average for solving
IS_RECORDING = True 

# Fine-tuning the EPSILON_DECAY parameters will lead to better results for 
# some environments and worse for others. As this code is a go at a 
# general player, it is neater to treat it as a global constant 
EPSILON_DECAY = 0.99

# Import basic libraries
import numpy as np


"""
Main loop for the game's initialization and runs
"""
def main():

    # Import gym and launch the game
    import gym
    env = gym.make(game_name)
    
    assert isinstance(env.action_space, gym.spaces.discrete.Discrete), (
        'env.action_space is not Discrete and is currently unsupported')
    assert isinstance(env.observation_space, gym.spaces.box.Box), (
        'env.observation_space is not continuous and is currently unsupported')
    assert len(env.observation_space.shape) == 1, (
        'env.observation_space is multi-dimensional and currently unsupported')
    if IS_RECORDING:
        env.monitor.start('results-' + game_name, force=True)
        
    # Parameters of the game and learning
    obs_space = env.observation_space.shape[0]
    architecture = [obs_space,
                    25 * obs_space, 
                    25 * obs_space,
                    env.action_space.n]
    
    # Create a gym instance and initialize the agent
    agent = Agent(env.action_space.n,
                  obs_space,
                  architecture) 
    reward, done, = 0.0, False
        
    
    # Start the game loop
    for episode in range(1, MAX_EPISODES + 1):
        obs, done = env.reset(), False
        action = agent.act(obs, reward, done, episode)
        
        while not done:
            
            # Un-comment to show the game on screen 
            #env.render()
            
            # Decide next action and feed the decision to the environment         
            obs, reward, done, _ = env.step(action)  
            action = agent.act(obs, reward, done, episode)
        
            
    # Save info and shut activities
    env.close()
    if IS_RECORDING:
        env.monitor.close()
    

"""
The general game player implements the Q-learning method with minibatches
"""
class Agent():

    def __init__(self, n_actions, obs_space, architecture):

        # Initialization of useful variables and constants
        self._N_ACTIONS = n_actions
        self._OBS_SPACE = obs_space 
        self._architecture = architecture
        self._nn = FeedForwardNeuralNetwork(architecture)
        
        # Hyperparameters of the training
        self._DISCOUNT_FACTOR = 0.99    # discount of future rewards
        self._TRAINING_PER_STAGE = 4
        self._MINIBATCH_SIZE = 128      
        self._REPLAY_MEMORY = 50000     

        # Exploration/exploitations parameters
        self._epsilon = 1.
        self._EPSILON_DECAY = EPSILON_DECAY
        self._EPISODES_PURE_EXPLORATION = 25
        self._MIN_EPSILON = 0.1

        # Define useful variables
        self._total_reward, self._list_rewards = 0.0, []
        self._last_action = np.zeros(self._N_ACTIONS)
        self._previous_observations, self._last_state = [], None
        self._LAST_STATE_IDX = 0
        self._ACTION_IDX = 1 
        self._REWARD_IDX = 2
        self._CURR_STATE_IDX = 3
        self._TERMINAL_IDX = 4
        
        
    def act(self, obs, reward, done, episode):
        
        self._total_reward += reward
        
        if done:
            self._list_rewards.append(self._total_reward)
            average = np.mean(self._list_rewards[-CONSECUTIVE_EPISODES:])
            print ('Episode', episode, 'Reward', self._total_reward, 
                   'Average Reward', round(average, 2))           
            self._total_reward = 0.0
            self._epsilon = max(self._epsilon * self._EPSILON_DECAY,
                                self._MIN_EPSILON)       
                  
        # Reshape the data
        current_state = obs.reshape((1, len(obs)))
        
        # Initialize last_state
        if self._last_state is None:
            self._last_state = current_state
            value_per_action = self._nn.predict(self._last_state)
            chosen_action_index = np.argmax(value_per_action)  
            self._last_action = np.zeros(self._N_ACTIONS)
            self._last_action[chosen_action_index] = 1
            return (chosen_action_index)

        # Store the last transition
        new_observation = [0 for _ in range(5)]
        new_observation[self._LAST_STATE_IDX] = self._last_state.copy()
        new_observation[self._ACTION_IDX] = self._last_action.copy()
        new_observation[self._REWARD_IDX] = reward
        new_observation[self._CURR_STATE_IDX] = current_state.copy()
        new_observation[self._TERMINAL_IDX] = done
        self._previous_observations.append(new_observation)
        self._last_state = current_state.copy()
            
        # If the memory is full, pop the oldest stored transition
        while len(self._previous_observations) >= self._REPLAY_MEMORY:
            self._previous_observations.pop(0)
        
        # Only train and decide after enough episodes of random play
        if episode > self._EPISODES_PURE_EXPLORATION:
  
            for _ in range(self._TRAINING_PER_STAGE):
                self._train()       
                
            # Chose the next action with an epsilon-greedy approach
            if np.random.random() > self._epsilon:
                value_per_action = self._nn.predict(self._last_state)
                #if episode > 150 and np.random.random() < 1e-1:
                #    print value_per_action
                chosen_action_index = np.argmax(value_per_action)  
            else:
                chosen_action_index = np.random.randint(0, self._N_ACTIONS)
        
        else:
            chosen_action_index = np.random.randint(0, self._N_ACTIONS)
    
        next_action_vector = np.zeros([self._N_ACTIONS])
        next_action_vector[chosen_action_index] = 1
        self._last_action = next_action_vector
          
        return (chosen_action_index)

    def _train(self):
        
        # Sample a mini_batch to train on
        permutations = np.random.permutation(
            len(self._previous_observations))[:self._MINIBATCH_SIZE] 
        previous_states = np.concatenate(
            [self._previous_observations[i][self._LAST_STATE_IDX]
            for i in permutations], 
            axis=0)
        actions = np.concatenate(
            [[self._previous_observations[i][self._ACTION_IDX]] 
            for i in permutations], 
            axis=0)
        rewards = np.array(
            [self._previous_observations[i][self._REWARD_IDX] 
            for i in permutations]).astype('float')
        current_states = np.concatenate(
            [self._previous_observations[i][self._CURR_STATE_IDX] 
            for i in permutations], 
            axis=0)
        done = np.array(
            [self._previous_observations[i][self._TERMINAL_IDX] 
            for i in permutations]).astype('bool')

        # Calculates the value of the current_states (per action)
        valueCurrentstates = self._nn.predict(current_states)
        
        # Calculate the empirical target value for the previous_states
        valuePreviousstates = rewards.copy()
        valuePreviousstates += ((1. - done) * 
                                self._DISCOUNT_FACTOR * 
                                valueCurrentstates.max(axis=1))

        # Run a training step
        self._nn.fit(previous_states,
                          actions, 
                          valuePreviousstates)


"""
Plain Feed Forward Neural Network
The chosen activation function is the Leaky ReLU function
"""
class FeedForwardNeuralNetwork:
    
    def __init__(self, layers):

        # NN variables
        self._generateNetwork(np.array(layers))


    def _generateNetwork(self, layers):
        """
        The network is implemented in Numpy
        Change this method if you wish to use a different library
        """
        
        self._ALPHA = 1e-2
        # Activation function used is the Leaky ReLU function
        self._activation = lambda x : x * (0.01*(x<0) + (x>=0))
        self._derive = lambda x : 0.01*(x<0) + (x>=0)
        
        # Initialization parameters
        INITIALIZATION_WEIGHTS = 0.1
        INITIALIZATION_BIAS = -0.001

        # Create the graph's architecture
        self._weights = []
        self._bias = []

        for i in range(layers.size - 1):
            weight = np.random.uniform(-INITIALIZATION_WEIGHTS, 
                                       INITIALIZATION_WEIGHTS,
                                       size=(layers[i], layers[i+1]))
            bias = INITIALIZATION_BIAS * np.ones((layers[i+1]))
            self._weights.append(weight)
            self._bias.append(bias)
            
    def _feedFwd(self, X):

        self._activation_layers = [np.atleast_2d(X)]
        
        for i in range(len(self._weights) - 1):
            self._activation_layers.append(self._activation(
                np.dot(self._activation_layers[-1], 
                       self._weights[i]) +
            self._bias[i]))
                
        # Last layer does not require the activation function
        self._activation_layers.append(
            np.dot(self._activation_layers[-1], 
                   self._weights[-1]) + 
            self._bias[-1])
        
        return(self._activation_layers[-1])          

    def _backProp(self, X, a, y):    
        
        # Calculate the delta vectors
        self._delta_layers = [a * (np.atleast_2d(y).T - self._feedFwd(X))]
        
        for i in range(len(self._activation_layers) - 2, 0, -1):
            self._delta_layers.append(
                np.dot(
                    self._delta_layers[-1],
                    self._weights[i].T) * 
                self._derive(self._activation_layers[i]))
            
        self._delta_layers.reverse()
        
        # Reduce the learning rate if the error grows eccessively
        if np.array([np.abs(delta).sum() 
                    for delta in self._delta_layers]).sum() > 1. / self._ALPHA:
            self._ALPHA /= 2.
            
        # Update the weights and bias vectors
        for i in range(len(self._weights)):
            self._weights[i] += self._ALPHA * np.dot(
                self._activation_layers[i].T,
                self._delta_layers[i])
            self._bias[i] += self._ALPHA * self._delta_layers[i].sum(axis=0)
                                       
                                         
    def predict(self, state):    
        return(self._feedFwd(state))
       
    def fit(self, valueStates, actions, valueTarget):                      
        self._backProp(valueStates, actions, valueTarget)


if __name__=="__main__":
   main()