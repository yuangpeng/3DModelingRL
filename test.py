from xml.etree.ElementTree import PI
import gym
import numpy as np
from stable_baselines3 import SAC
import chimera
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    # env = gym.make("Mesh-v0")
    # check_env(env)

    # obs = env.reset()
    # reward = -1
    # i = 0
    # while i != 10:
    #     print(f"state : {obs}, reward : {reward}")
    #     action = np.random.uniform(-1, 1, size=(162,))
    #     obs, reward, done, _ = env.step(action)
    #     env.output_result(f"res{i}", "./")
    #     i += 1
    #     if done:
    #         brea

    env = gym.make("Prim-v0")

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("sac_pendulum")

    del model  # remove to demonstrate saving and loading

    model = SAC.load("sac_pendulum")

    obs = env.reset()
    i = 0
    while i != 10:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.output_result(f"res{i}", "./")
        i += 1
        if done:
            obs = env.reset()
