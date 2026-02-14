import game
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output
import imageio

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.gelu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.gelu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.gelu1(self.fc1(x))
        x = self.gelu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

        

class Agent:
    def __init__(self, num_epoches=100, gamma=0.99, lr=1e-3):
        self.gamma = gamma
        self.game = game.FallingBlocksGame()
        self.initial_state = self.game.reset()
        self.policy = PolicyNetwork(len(self.initial_state), 3)
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.num_epoches = num_epoches
        self.loss_history = []
        self.score_history = []
        self.length_history = []

    def train_step(self):
        self.game.reset()
        running = True
        states = []
        rewards = []
        log_probs = []
        entropies = []

        G = 0
        steps = 0
        while running:
            steps += 1
            probs = self.policy(torch.tensor(self.game._get_state()).float())
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))
            entropies.append(m.entropy())
            state, reward, done = self.game.step(action.item())
            states.append(state)
            rewards.append(reward)

            if done:
                break
        
        states = torch.tensor(states)
        rewards = torch.tensor(rewards)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)
        G = np.zeros_like(rewards)
        G[-1] = rewards[-1]
        for i in range(2, G.shape[0] + 1):
            G[-i] = G[-i + 1] * self.gamma + rewards[-i]

        G = torch.tensor(G)
        G = (G - G.mean()) / (G.std() + 1e-9)

        loss = -torch.mean(log_probs_tensor * G) - 0.01 * entropies_tensor.mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss, steps
    

    def draw_stats(self, window=50): # window — за сколько эпизодов усредняем
        clear_output(wait=True)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Лосс (обычно его не усредняют, чтобы видеть выбросы)
        axs[0].plot(self.loss_history, color='red', alpha=0.3) # Бледный основной
        axs[0].set_title("Loss")

        # 2. Total Score
        axs[1].plot(self.score_history, color='blue', alpha=0.3) # Бледный оригинал
        if len(self.score_history) >= window:
            # Считаем среднее
            means = np.convolve(self.score_history, np.ones(window)/window, mode='valid')
            axs[1].plot(range(window-1, len(self.score_history)), means, color='darkblue', linewidth=2, label=f'SMA {window}')
        axs[1].set_title("Total Score")
        axs[1].legend()

        # 3. Episode Length
        axs[2].plot(self.length_history, color='green', alpha=0.3) # Бледный оригинал
        if len(self.length_history) >= window:
            # Считаем среднее
            means = np.convolve(self.length_history, np.ones(window)/window, mode='valid')
            axs[2].plot(range(window-1, len(self.length_history)), means, color='darkgreen', linewidth=2, label=f'SMA {window}')
        axs[2].set_title("Episode Length")
        axs[2].legend()

        plt.show()

    def train(self):
        for epoch in range(self.num_epoches):
            loss, length = self.train_step()
            
            self.loss_history.append(loss.item())
            self.score_history.append(self.game.score)
            self.length_history.append(length)

            if epoch % 5 == 0:
                self.draw_stats()

    def save_inference_gif(self, filename="agent_play.gif", fps=30):
        self.policy.eval()  # Переводим сеть в режим оценки
        self.game.reset()
        frames = []
        running = True
        
        with torch.no_grad(): # Выключаем градиенты
            while running:
                # Получаем картинку/кадр из игры (убедись, что в game есть метод render или аналогичный)
                # Если FallingBlocksGame возвращает массив отрисовки:
                frame = self.game.render() 
                frames.append(frame)
                
                # Выбираем лучшее действие (не сэмплируем, а берем argmax)
                state = torch.tensor(self.game._get_state()).float()
                probs = self.policy(state)
                action = torch.argmax(probs).item()
                
                _, _, done = self.game.step(action)
                
                if done:
                    break
                    
        # Сохраняем собранные кадры
        imageio.mimsave(filename, frames, fps=fps)
        print(f"GIF сохранен как {filename}")
        self.policy.train() # Возвращаем сеть в режим обучения


            
