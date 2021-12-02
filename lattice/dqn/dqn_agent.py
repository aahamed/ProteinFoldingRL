import gym
import argparse
from gym_lattice.envs import Lattice2DEnv

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import time

def run_dqn_agent( args ):
    # seed
    set_random_seed(args.seed)

    # Create environment
    # env = gym.make('LunarLander-v2')
    seq = 'HHPPHHPPHH' # Our input sequence
    seq = seq.upper()
    env = Lattice2DEnv(seq)

    # Instantiate the agent
    model = DQN('MlpPolicy', env, verbose=1,
        exploration_fraction=0.2, exploration_final_eps=0.1,
        tensorboard_log='./tensorboard')

    start = time.time()
    model.learn(total_timesteps=int(args.num_timesteps),
            tb_log_name=args.save_name)
    end = time.time()
    # Save the agent
    model.save(args.save_name)
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = DQN.load(args.save_name, env=env)

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

    print( f'final energy: {env._compute_free_energy(env.state)}' )
    print( f"Total time needed to train: {end-start:.2f}" )

def main():
    parser = argparse.ArgumentParser(description='DQN Agent on Lattice Environment')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num-timesteps', type=float, default=2e4,
                        help='number of timesteps')
    parser.add_argument('--save-name', type=str, default='dqn_lattice',
                        help='save name for trained model')
    args = parser.parse_args()
    run_dqn_agent( args )

if __name__ == '__main__':
    main()
