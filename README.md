# General OpenAI gym player
**Simple feed forward neural net that solves [OpenAI gym][5] environments via Q-learning**

### Quick start
Download the code, assign to game_name the name of environment you wish to run, and let the script learn how to solve it.   
Note the code only works for environments with discrete action space and continuous observation space.

Results can be found on [https://gym.openai.com/users/FlankMe][4]

### Implementation of the neural net
As the script relies on a plain feed-forward neural net, really I should have coded it in Numpy.  

Yet, I lazily chose to let TensorFlow do the heavy lifting and to enjoy the convenience of the readily-available Adam Optimizer.   
I may publish a Numpy (stochastic gradient descent) implementation at a later stage.  

**EDIT: Numpy implementation is now available in the same folder. File name is `GeneralGymPlayerWithNP.py`.**

<img src="https://github.com/FlankMe/general-gym-player/blob/master/Animations/CartPole-v0.gif" width="30%" />
<img src="https://github.com/FlankMe/general-gym-player/blob/master/Animations/Acrobot-v0.gif" width="30%" /> 
<img src="https://github.com/FlankMe/general-gym-player/blob/master/Animations/LunarLander-v2.gif" width="30%" />

### Requirements
* **Python 3**. I recommend this version as it's the only one I found compatible with the below libraries;
* **TensorFlow**, I only managed to install it on my Mac. Download it from [here][2];
* **Gym**, open-source collection of test problems for reinforcement learning algorithms. Details on how to download it can be found [here][3]. 

[2]: https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html 
[3]: https://gym.openai.com/docs
[4]: https://gym.openai.com/users/FlankMe
[5]: https://gym.openai.com
