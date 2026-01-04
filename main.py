import gymnasium as gym
from minesweeper_env_gymnasium import MinesweeperEnv
import pygame
from dqn import DQNAgent
import torch
from datetime import datetime

#env = MinesweeperEnv(height=8, width=8, num_mines=10)
#agent = DQNAgent(env)

def save_model(agent, episodes, best_reward, wins):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  #nt 20250123_143052
    win_rate = (wins / episodes) * 100
    filename = f"minesweeper_ep{episodes}_wr{win_rate:.0f}_best{best_reward:.0f}_{timestamp}.pth"
    
    torch.save({
        'model_state': agent.policy_net.state_dict(),
        'episodes': episodes,
        'best_reward': best_reward,
        'wins': wins,
        'win_rate': win_rate,
        'epsilon': agent.epsilon
    }, filename)
    
    print("f\nSalvestatud: {filename}")
    return filename

def train(episodes=1000):
    env = MinesweeperEnv(6, 6, 5)
    agent = DQNAgent(env)
    best_reward = -float('inf')  #parim skoor 
    wins = 0  #mitu võitu
    best50 = 0
    recent_rewards = []  
    #hiljem salvestatud mudeli failinime jaoks
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask) #annab ette seisu ja kinni/lahti ruudud
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
                
            #leiab uue seisu kui käik tehtud
            agent.memory.push(state, action, reward, next_state, done)
            #jätab meelde
            
            loss = agent.update(batch_size=32)
            #õpib 32 kogemuse pealt ja saab teada kui vale ta ennustused on
            
            state = next_state
            total_reward+= reward
            if total_reward > best_reward:
                best_reward = total_reward
        recent_rewards.append(total_reward) #
        if len(recent_rewards) > 50: #
            recent_rewards.pop(0) #
        if terminated and reward > 0:
            wins += 1

        agent.epsilon = max(agent.epsilon_min, agent.epsilon*agent.epsilon_decay) # mida rohkem õpib seda vähem kondab
        
        if episode % 100 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            #iga 10 episoodi õpetab staatilisele mudelile, mida väiksem, seda ebastabiilsem, mida suurem, seda aegunum eesmärk
            
        if episode % 50 == 0:
            win_rate = wins / 50 * 100; wins=0 
            avg_recent = (sum(recent_rewards) / len(recent_rewards))/env.rewards["safe"]
            best50 = max(recent_rewards)/env.rewards["safe"]
            print(f"Episode {episode}, Best50: {best50:.2f}, Avg50: {avg_recent:.2f}, Win Rate: {win_rate:.1f}%, Epsilon: {agent.epsilon:.3f}")
            print(f"  Buffer size: {len(agent.memory)}, Last reward: {total_reward/env.rewards['safe']:.1f}")
            #logging, aitab silma peal hoida sellel kui hästi läheb
    return agent, best_reward, wins
        

#pealtvaatamine :p

if __name__ == "__main__":
    episodes = 10000
    trained_agent, best_reward, wins = train(episodes=episodes)
    #save_model(trained_agent, episodes, best_reward, wins)
    filename = save_model(trained_agent, episodes, best_reward, wins)  
