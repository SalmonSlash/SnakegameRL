import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

BLOCK_SIZE = 20

class Direction:
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __hash__(self):
        return hash((self.x, self.y))

class SillySnakeGameAi:
    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.food = None
        self.placeFood()

        self.frameIteration = 0
        return self._getState()

    def placeFood(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            foodPoint = Point(x, y)
            if foodPoint not in self.snake:
                self.food = foodPoint
                break

    def _getState(self):

        head = self.head

        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        danger_straight = False
        danger_right = False
        danger_left = False

        if self.direction == Direction.RIGHT:
            danger_straight = self.isCollision(point_right)
            danger_right = self.isCollision(point_down)
            danger_left = self.isCollision(point_up)
        elif self.direction == Direction.LEFT:
            danger_straight = self.isCollision(point_left)
            danger_right = self.isCollision(point_up)
            danger_left = self.isCollision(point_down)
        elif self.direction == Direction.UP:
            danger_straight = self.isCollision(point_up)
            danger_right = self.isCollision(point_right)
            danger_left = self.isCollision(point_left)
        elif self.direction == Direction.DOWN:
            danger_straight = self.isCollision(point_down)
            danger_right = self.isCollision(point_left)
            danger_left = self.isCollision(point_right)

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(self.direction == Direction.LEFT),
            int(self.direction == Direction.RIGHT),
            int(self.direction == Direction.UP),
            int(self.direction == Direction.DOWN),

            int(self.food.x < head.x),
            int(self.food.x > head.x),
            int(self.food.y < head.y),
            int(self.food.y > head.y)
        ]

        return np.array(state, dtype=int)

    def playStep(self, action):

        self.frameIteration += 1
        self.moveSnake(action)

        reward = 0
        gameOver = False

        if self.isCollision() or self.frameIteration > 100 * len(self.snake):
            gameOver = True
            reward = -10
            return reward, gameOver, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.placeFood()
        else:

            self.snake.pop()
        return reward, gameOver, self.score

    def isCollision(self, p=None):
        if p is None:
            p = self.head

        if p.x >= self.w or p.x < 0 or p.y >= self.h or p.y < 0:
            return True

        if p in self.snake[1:]:
            return True
        return False

    def moveSnake(self, action):

        move = np.argmax(action)
        if move == 0:
            self.direction = Direction.UP
        elif move == 1:
            self.direction = Direction.RIGHT
        elif move == 2:
            self.direction = Direction.DOWN
        elif move == 3:
            self.direction = Direction.LEFT

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        self.snake.insert(0, self.head)

class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, hiddenSize)
        self.linear3 = nn.Linear(hiddenSize, outputSize)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def trainStep(self, state, action, reward, next_state, done):
        stateTensor = torch.tensor(state, dtype=torch.float)
        actionTensor = torch.tensor(action, dtype=torch.long)
        rewardTensor = torch.tensor(reward, dtype=torch.float)
        nextStateTensor = torch.tensor(next_state, dtype=torch.float)

        if len(stateTensor.shape) == 1:

            stateTensor     = torch.unsqueeze(stateTensor, 0)
            actionTensor    = torch.unsqueeze(actionTensor, 0)
            rewardTensor    = torch.unsqueeze(rewardTensor, 0)
            nextStateTensor = torch.unsqueeze(nextStateTensor, 0)
            done = (done,)

        pred = self.model(stateTensor)
        target = pred.clone()

        for i in range(len(done)):
            Q_new = rewardTensor[i]
            if not done[i]:
                Q_new = rewardTensor[i] + self.gamma * torch.max(self.model(nextStateTensor[i]))
            target[i][torch.argmax(actionTensor[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100000)

        self.model = LinearQNet(11, 256, 4)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def getState(self, game):
        return game._getState()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainLongMemory(self, batch_size=1000):
        if len(self.memory) < batch_size:
            sample = self.memory
        else:
            sample = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.trainStep(states, actions, rewards, next_states, dones)

    def trainShortMemory(self, state, action, reward, next_state, done):
        self.trainer.trainStep(state, action, reward, next_state, done)

    def getAction(self, state):

        self.epsilon = max(0.01, 0.8 * (0.995 ** self.n_games))
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train(n_episodes):
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SillySnakeGameAi()

    while agent.n_games < n_episodes:
        state_old = agent.getState(game)

        final_move = agent.getAction(state_old)

        reward, done, score = game.playStep(final_move)
        state_new = agent.getState(game)

        agent.trainShortMemory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.trainLongMemory()

            if score > record:
                record = score
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            if agent.n_games % 10 == 0 or score == record:
                print(f"Game {agent.n_games}, Score: {score}, Best: {record}, Mean: {mean_score:.2f}")

    torch.save(agent.model.state_dict(), "doUperfec.pth")
    print("Model saved to", "doUperfec.pth")
    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.trainer.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'episode': n_episodes
    }

    torch.save(checkpoint, "doUperfec.pthcheckpoint.pth")
    print("Checkpoint saved.")

if __name__ == "__main__":
    train(n_episodes=500)

















