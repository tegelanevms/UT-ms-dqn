import gymnasium as gym
from minesweeper_env_gymnasium import MinesweeperEnv
import pygame
from dqn import DQNAgent
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
epcount =1000000
#env = MinesweeperEnv(height=8, width=8, num_mines=10)
#agent = DQNAgent(env)

def save_model(agent, episodes, best_reward, wins):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  #nt 20250123_143052
    filename = f"minesweeper_ep{episodes}_{timestamp}.pth"
    
    torch.save({
        'model_state': agent.policy_net.state_dict(),
        'episodes': episodes,
        'wins': wins,
        'epsilon': agent.epsilon
    }, filename)
    
    f"\nSalvestatud: {filename}"
    return filename

def train(episodes=epcount, agent=None):
    writer = SummaryWriter()
    env = MinesweeperEnv(6, 6, 5)
    if agent is None:
        agent = DQNAgent(env)
    best_reward = -float('inf')  #parim skoor 
    wins = 0  #mitu võitu
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
            
            state = next_state
            total_reward+= reward
            if total_reward > best_reward:
                best_reward = total_reward
        agent.update()
        agent.update()
        writer.add_scalar('Pühkur/Iga episoodi tulemus üldse', total_reward, episode)
        #2 korda for funsies, õpib tsipa kiiremini
        recent_rewards.append(total_reward) #
        if len(recent_rewards) > 1000: #
            recent_rewards.pop(0) #
        if terminated and reward > 0:
            wins += 1

        agent.epsilon = max(agent.epsilon_min, agent.epsilon*agent.epsilon_decay) # mida rohkem õpib seda vähem kondab
          
        if episode % 1000 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            win_rate = wins / 1000 * 100; wins=0 
            avg_recent = (sum(recent_rewards) / len(recent_rewards))/env.rewards["safe"]
            print(f"Episode {episode}, Avg of : {avg_recent:.2f}, Win Rate: {win_rate:.1f}%, Epsilon: {agent.epsilon:.3f}")
            print(f"  Buffer size: {len(agent.memory)}")
            writer.add_scalar('Pühkur/Võidumäär 1k', win_rate, episode)
            writer.add_scalar('Pühkur/Tulem 1k', avg_recent, episode)
            #logging, aitab silma peal hoida sellel kui hästi läheb
        if episode % 5000 == 0 and episode > 0:
            save_model(agent, episode, best_reward, wins)
    writer.close()
    return agent, best_reward, wins
        


if __name__ == "__main__":
    episodes = epcount
    #trained_agent, best_reward, wins = train(episodes=episodes)
    #save_model(trained_agent, episodes, best_reward, wins)
    #filename = save_model(trained_agent, episodes, best_reward, wins)  

    trained_agent, best_reward, wins = train(episodes=1000000)
    save_model(trained_agent, "1m", best_reward, wins)