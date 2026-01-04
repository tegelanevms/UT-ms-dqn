import gymnasium as gym
from minesweeper_env_gymnasium import MinesweeperEnv
import pygame
from dqn import DQNAgent
import torch
import time

def load_and_play(model_path, episodes=5, delay=0.3):
    # env peab klappima originaaliga
    env = MinesweeperEnv(6, 6, 5)  
    agent = DQNAgent(env)
    
    
    checkpoint = torch.load(model_path)
    agent.policy_net.load_state_dict(checkpoint['model_state'])
    agent.epsilon = 0.0  # ta teeb aint neid käike mis on parimad
    
    print(f"Loaded model from {model_path}")
    print(f"Training stats: Episodes: {checkpoint['episodes']}, Win Rate: {checkpoint['win_rate']:.1f}%")
    print(f"Best reward: {checkpoint['best_reward']:.1f}\n")
    
    wins = 0
    total_rewards = []
    
    for episode in range(episodes):
        state, info = env.reset()
        env.render()  
        
        total_reward = 0
        done = False
        step = 0
        
        print(f"\n=== Episode {episode + 1}/{episodes} ===")
        
        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask)
            
            x = action // env.width
            y = action % env.width
            print(f"Step {step}: Clicking ({x}, {y})")
            
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
            step += 1
            
            
            env.render()
            time.sleep(delay)
            
            if done:
                if terminated and reward > 0:
                    wins += 1
                    print(f"win Total reward: {total_reward:.1f}")
                    time.sleep(10)
                else:
                    print(f"loss Total reward: {total_reward:.1f}")
                
                total_rewards.append(total_reward)
                time.sleep(2)  
    
    env.close()
    
    print(f"\n{'='*50}")
    print(f"Results over {episodes} episodes:")
    print(f"Wins: {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"Avg reward: {sum(total_rewards)/len(total_rewards):.1f}")
    print(f"Best reward: {max(total_rewards):.1f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    model_path = "minesweeper_ep10000_wr0_best193_20260104_154346.pth"
    
    load_and_play(model_path, episodes=100, delay=0.1)