import gym
from pulling_env import Pulling2DEnv

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
# env = gym.make('LunarLander-v2')
seq = 'HHPPHH' # Our input sequence
seq = seq.upper()
env = Pulling2DEnv(seq)

# Instantiate the agent
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(2e3))
# Save the agent
model.save("A2C_pulling")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = A2C.load("A2C_pulling", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
i = 0
while not env.done or i > 1000:
	action, _states = model.predict(obs, deterministic=True)
	obs, rewards, dones, info = env.step( action )
	env.render()
	i += 1
