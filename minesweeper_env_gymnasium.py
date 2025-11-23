import gymnasium as gym
from gymnasium import spaces
import numpy as np
from random import randrange
import pygame
import pygame.freetype

class MinesweeperEnv(gym.Env):

    # Enhanced minesweeper environment for DQN training
    # STATE
    #   Multi-channel observation:
    #   - Channel 0: cell values (-1=hidden, 0-8=revealed numbers)
    #   - Channel 1: revealed mask (0=hidden, 1=revealed)
    # MAP
    #   the map is unobservable to the agent where 0 is non-mine, 1 has a mine
    # ACTIONS
    #   action is Discrete(height*width) - flattened cell index to reveal
    # REWARD SHAPING
    #   +1 per safe cell revealed, -10 for mines, +10 for win, -0.2 for invalid
    # ACTION MASKING
    #   get_action_mask() returns valid actions (only hidden cells)

    def __init__(self, height=8, width=8, num_mines=10):
        # Multi-channel observation: [cell_values, revealed_mask]
        self.observation_space = spaces.Box(
            low=-1, high=8, 
            shape=(height, width, 2), 
            dtype=np.float32
        )
        
        # Flattened action space - easier for DQN
        self.action_space = spaces.Discrete(height * width)

        self.height = height
        self.width = width
        self.num_mines = num_mines
        self.rewards = {
            "safe": 1,
            "fail": -10,
            "win": 10,
            "invalid": -0.2
        }
        
        self.map = np.array([[False]*width for _ in range(height)])
        self.state = np.zeros((height, width), dtype=np.float32) - 1
        self.revealed_mask = np.zeros((height, width), dtype=np.float32)
        self.step_cntr = 0
        self.step_cntr_max = (height*width-num_mines)*2

        self.block_size = 25
        self.window_height = self.block_size * height
        self.window_width = self.block_size * width
        self.map = None
        self.generate_mines()
        
        self.screen = None

    def generate_mines(self):
        self.map = np.array([[False]*self.width for _ in range(self.height)])
        for _ in range(self.num_mines):
            x = randrange(self.height)
            y = randrange(self.width)
            while self.map[x,y]:
                x = randrange(self.height)
                y = randrange(self.width)
            self.map[x,y] = True

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.generate_mines()
        self.step_cntr = 0
        self.state = np.zeros((self.height, self.width), dtype=np.float32) - 1
        self.revealed_mask = np.zeros((self.height, self.width), dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        """Returns multi-channel observation: [state, revealed_mask]"""
        return np.stack([self.state, self.revealed_mask], axis=-1)

    def get_action_mask(self):
        """Returns binary mask of valid actions (1=valid, 0=invalid)"""
        return (self.revealed_mask == 0).flatten().astype(np.float32)

    def get_num_opened(self):
        return int(np.sum(self.revealed_mask))

    def get_num_surr(self, x, y):
        count = 0
        for i in range(max(0,x-1), min(self.height,x+2)):
            for j in range(max(0,y-1), min(self.width,y+2)):
                if not (i==x and j==y):
                    if self.map[i,j]:
                        count += 1
        return count

    def update_state(self, x, y):
        num_surr = self.get_num_surr(x,y)
        self.state[x,y] = num_surr
        self.revealed_mask[x,y] = 1
        
        # Cascade reveal if no surrounding mines
        if num_surr==0:
            for i in range(max(0,x-1), min(self.height,x+2)):
                for j in range(max(0,y-1), min(self.width,y+2)):
                    if (not (i==x and j==y)) and self.revealed_mask[i,j]==0:
                        self.update_state(i,j)

    def step(self, action):
        # Convert flattened action to coordinates
        action = int(action)
        x = action // self.width
        y = action % self.width
        
        info = self._get_info()
        
        # Check for truncation (max steps)
        if self.step_cntr >= self.step_cntr_max:
            return self._get_obs(), 0, False, True, info
        
        self.step_cntr += 1
        
        # Invalid action - clicking already revealed cell
        if self.revealed_mask[x,y] == 1:
            return self._get_obs(), self.rewards["invalid"], False, False, info
        
        # Hit a mine - terminated
        if self.map[x][y]:
            self.revealed_mask[x,y] = 1
            return self._get_obs(), self.rewards["fail"], True, False, info
        
        # Valid reveal
        num_opened = self.get_num_opened()
        self.update_state(x,y)
        new_num_opened = self.get_num_opened()
        
        cells_revealed = new_num_opened - num_opened
        reward = self.rewards["safe"] * cells_revealed
        
        # Won the game - terminated
        if new_num_opened == self.height*self.width - self.num_mines:
            reward += self.rewards["win"]
            return self._get_obs(), reward, True, False, info
        
        # Normal step
        return self._get_obs(), reward, False, False, info

    def drawGrid(self):
        for y in range(0, self.window_width, self.block_size):
            for x in range(0, self.window_height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                num = int(self.state[int(x/self.block_size),int(y/self.block_size)])
                if num==-1:
                    pygame.draw.rect(self.screen, (255,255,255), rect, 1)
                else:
                    color = (250, 250-num*30, 250-num*30)
                    pygame.draw.rect(self.screen, color, rect)
                    text = self.font.get_rect(str(num))
                    text.center = rect.center
                    self.font.render_to(self.screen,text.topleft,str(num),(0,0,0))
        pygame.display.update()

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.font = pygame.freetype.SysFont(pygame.font.get_default_font(), 13)
        self.screen.fill((0,0,0))
        self.drawGrid()

    def _get_info(self):
        return {
            "map": self.map,
            "action_mask": self.get_action_mask(),
            "cells_revealed": self.get_num_opened(),
            "cells_remaining": self.height * self.width - self.num_mines - self.get_num_opened()
        }
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()