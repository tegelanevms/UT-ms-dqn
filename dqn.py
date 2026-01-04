import torch_directml
import torch
#vajab erilist torchi versiooni
#pip install torch-directml
import torch.nn as nn
import numpy as np
from collections import deque
import random as py_random

device = torch_directml.device()
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, height=8, width=8, n_actions=64):
        super().__init__()
        #sequential et ta läbiks eri filtrid ühekaupa järjest?
        #CNN-id vajalikud kuna ruumiline asukoht loeb minesweeperis vms
        self.conv = nn.Sequential(
            #2 input (cell + revealed mask), 32 output, 3x3 filter, padding hoiab lauasuuruse samaks?
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(height * width * 128, 1024), #lineaarsete kihtide jaoks peab conv-ist tulevad 4D tensorid lapikuks tegema vms, see rida surub kokku 512ks
            nn.ReLU(), #paneb kihid omavahel tööle or sum shi
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions) #DQN-i jaosk need Q väärtused antud kihtidest
        )
    
    def forward(self, x):
    # x shape: (batch, height, width, channels (alati 2))
        x = x.permute(0, 3, 1, 2)  # -> (batch, 2, height, width) (pmst keerab ringi need et oleks lihtsam)
        x = self.conv(x) #lööb sealt CNN-ist läbi asjad
        x = x.flatten(1) #vt fc rida 1
        x = self.fc(x) #fully connected jamps
        return x
        
        
class ReplayBuffer: #mälu lis, et saaks võtta suvalise testbatchi
    def __init__(self, capacity=10000):
        #deque võtab esimese ära kui list täis saab
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        #jätab ühe käigu/kogemuse meelde    
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = py_random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch) #unzipimine pmst
        return(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        #hetkene buffersize
        return(len(self.buffer))
        
class DQNAgent:
    def __init__(self, env, lr=0.0001, gamma=0.99):
        self.env = env
        self.gamma = gamma #kui olulised on järgmised sammud
        
        #policy õpib igal sammul, target iga N sammul
        self.policy_net = DQN(env.height, env.width, env.action_space.n).to(device) #aktiivselt õppiv tegelane
        self.target_net = DQN(env.height, env.width, env.action_space.n).to(device) #staatiline tagavara
        self.target_net.load_state_dict(self.policy_net.state_dict()) #pmst võtab need kalduvused mis policyl on targetile ehk ss alguses identsed
        
        self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.0001, momentum=0.9)
        
        #see osa goofballist mis suvaliselt ringi klõpsib
        self.epsilon = 1.0 #alustab 100% suvaliselt
        self.epsilon_min = 0.005 #lõpetab locked in
        self.epsilon_decay = 0.995
        # 1-epsilon peaks olema parim käik vist
        
        self.memory = ReplayBuffer(capacity=50000)  
        
    def select_action(self, state, action_mask):
        #väga epsiloniahne taku
        #state on lis mis ta näeb
        #action_mask on need vabad/kinni ruudud
        if py_random.random() < self.epsilon:
            #läheb suvaliselt uitama inshallah
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions)
            #kui ta otsustab uitama minna leiab kõik kohad kus on avamata ruut ja valib lampi ühe
            
            #väike insta reels break tulis sisse teha im so lost
            
        else: #i.e. kui ei lähe uitama
            with torch.no_grad(): #uhhhh midagi midagi ei jälgi gradiente et olla kiirem
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device) #unsqueeze annab batch dimensiooni (ei ole kindel mis) juurde ja ss to(device) saadab protsessorile

                q_values = self.policy_net(state_t).squeeze().cpu().numpy() #hr õppija uurib ja mõtiskleb, squeeze võtab batch dimensiooni ära, tagasi numpyks et seda maski kasutada
                
                q_values[action_mask == 0] = -np.inf #eemaldab ebasobivad käigud kuna argmax ei võta eales negatiivset lõputust
                
                return np.argmax(q_values) #parim käik
            
    def update(self, batch_size=32):
        #treeningloogika
        #vau see osa kus see D (deep) DQN nimest tuleb ju !!!!!!!
        
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size) #leiab 32 kogemust 
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        #saab väärtused, tensoriteks et gpu/cpu saaks nendega mõistatada
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)) #leiab iga kogemuse pealt selle sammu mis ta tegi
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
            #bellmani võrrand
            
            # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        #leiab kui kaugel praegune Q on sellest mida me tahame
        
        self.optimizer.zero_grad() #eemaldab vanad gradiendid
        loss.backward() #uus gradient
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step() #uued kalduvused
        
        return loss.item()  