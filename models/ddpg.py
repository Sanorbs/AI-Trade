import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

def DDPGAgent(multi_stock_env, num_episodes):
    models_folder = 'saved_models'
    rewards_folder = 'saved_rewards'

    env = DummyVecEnv([lambda: multi_stock_env])
    
    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

    # Hyper parameters
    GAMMA = 0.99
    TAU = 0.001
    BATCH_SIZE = 16
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_LEARNING_RATE = 0.001
    BUFFER_SIZE = 50000

    print("\nRunning DDPG Agent (SB3)...\n")
    model = DDPG(
        "MlpPolicy",
        env,
        gamma=GAMMA,
        tau=TAU,
        batch_size=BATCH_SIZE,
        learning_rate=ACTOR_LEARNING_RATE,  # SB3 uses a single learning_rate param
        buffer_size=BUFFER_SIZE,
        action_noise=action_noise,
        verbose=1
    )
    model.learn(total_timesteps=num_episodes * 200)  # 200 steps per episode (adjust as needed)
    model.save(f'{models_folder}/rl/ddpg_s3')

    del model
    
    model = DDPG.load(f'{models_folder}/rl/ddpg_s3')
    obs = env.reset()
    portfolio_value = []

    for e in range(num_episodes):
        done = False
        episode_reward = 0
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {episode_reward:.2f}")
        portfolio_value.append(round(episode_reward, 3))

    # save portfolio value for each episode
    np.save(f'{rewards_folder}/rl/ddpg_s3.npy', portfolio_value)

    print("\nDDPG Agent run complete and saved!")

    a = np.load(f'./saved_rewards/rl/ddpg_s3.npy')

    print(f"\nCumulative Portfolio Value Average reward: {a.mean():.2f}, Min: {a.min():.2f}, Max: {a.max():.2f}")
    plt.plot(a)
    plt.title("Portfolio Value Per Episode (DDPG, SB3)")
    plt.ylabel("Portfolio Value")
    plt.xlabel("Episodes")
    plt.show()