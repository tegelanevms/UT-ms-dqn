import gymnasium as gym
from minesweeper_env_gymnasium import MinesweeperEnv
import pygame
from dqn import mineqn

"""
env = MinesweeperEnv()
env.reset()
done = False
while not done:
    env.render()
    
    # Wait for any key press
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                waiting = False
                print(env._get_obs())
            if event.type == pygame.QUIT:
                env.close()
                exit()
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Reward: {reward}, Done: {done}")
env.close() """


def train():
    env = MinesweeperEnv(8, 8, 10)
    agent = mineqn(env)
    # ... training loop

if __name__ == "__main__":
    train()