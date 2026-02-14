import pygame
import random
import numpy as np

# Константы
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BLOCK_SIZE = 40
PLAYER_SIZE = 40
FPS = 60
MAX_BLOCKS_WATCH = 5 

class FallingBlocksGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Falling Blocks Survival")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.player_x = SCREEN_WIDTH // 2 - PLAYER_SIZE // 2
        self.player_y = SCREEN_HEIGHT - PLAYER_SIZE - 10
        self.blocks = []  # Список элементов [x, y]
        self.score = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        Возвращает вектор состояния:
        [player_x, block1_x, block1_y, block2_x, block2_y, ...]
        Все значения нормализованы от 0 до 1 для стабильности обучения.
        """
        # Нормализованная позиция игрока
        state = [self.player_x / SCREEN_WIDTH]
        
        # Сортируем блоки по близости к игроку (по Y)
        sorted_blocks = sorted(self.blocks, key=lambda b: b[1], reverse=True)
        
        for i in range(MAX_BLOCKS_WATCH):
            if i < len(sorted_blocks):
                # Добавляем координаты блока (нормализованные)
                state.append(sorted_blocks[i][0] / SCREEN_WIDTH)
                state.append(sorted_blocks[i][1] / SCREEN_HEIGHT)
            else:
                # Если блоков меньше лимита, заполняем "безопасными" значениями
                state.extend([0.0, 0.0]) 
        
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        action: 0 - Left, 1 - Stay, 2 - Right
        """
        if self.done:
            return self.reset(), 0, True

        # 1. Движение игрока
        speed = 10
        if action == 0 and self.player_x > 0:
            self.player_x -= speed
        elif action == 2 and self.player_x < SCREEN_WIDTH - PLAYER_SIZE:
            self.player_x += speed

        # 2. Логика блоков
        if random.random() < 0.08:
            self.blocks.append([random.randint(0, SCREEN_WIDTH - BLOCK_SIZE), -BLOCK_SIZE])

        for block in self.blocks[:]:
            block[1] += 7  # Скорость падения
            
            # Коллизия
            if (self.player_y < block[1] + BLOCK_SIZE and 
                self.player_y + PLAYER_SIZE > block[1] and
                self.player_x < block[0] + BLOCK_SIZE and
                self.player_x + PLAYER_SIZE > block[0]):
                self.done = True

            if block[1] > SCREEN_HEIGHT:
                self.blocks.remove(block)
                self.score += 1

        # 3. Награда
        # +0.1 за каждый выживший шаг, -10 за столкновение
        reward = 0.1 if not self.done else -10
        
        return self._get_state(), reward, self.done

    def render(self):
        # Отрисовку оставляем только для визуального контроля (можно выключать при обучении)
        self.screen.fill((30, 30, 30)) 
        pygame.draw.rect(self.screen, (0, 200, 255), (self.player_x, self.player_y, PLAYER_SIZE, PLAYER_SIZE))
        for block in self.blocks:
            pygame.draw.rect(self.screen, (255, 80, 80), (block[0], block[1], BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.display.flip()
        self.clock.tick(FPS)

if __name__ == "__main__":
    game = FallingBlocksGame()
    running = True
    
    while running:
        action = 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: action = 0
        if keys[pygame.K_RIGHT]: action = 2
        
        state, reward, done = game.step(action)
        game.render()
        
        if done:
            print(f"Final Vector State: {state}") # Посмотри, что видит агент
            game.reset()

    pygame.quit()