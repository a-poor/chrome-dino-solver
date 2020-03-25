"""
dino_solver.py

by Austin Poor



NOTES:
– Run for longer to see how it changes
– can you scale the inputs? (how would you handle)
– use more/less layers? to improve training?


– run multiple training sessions simultaneously
– run training headless in the cloud?

"""



import os
import time
from collections import deque

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout

URL = 'chrome://dino/'
CSS_RUNNER_CONTAINER = '.runner-container'

dt_fmt = time.strftime("%Y%m%d.%H%M%S")

class DinoGame:
    URL = 'chrome://dino/'
    CSS_RUNNER_CONTAINER = '.runner-container'

    FN = """return (function () {
            const results = [];
            const runn = new Runner();
            results.push(runn.crashed? 1.0 : 0.0);
            results.push(runn.distanceMeter.getActualDistance(runn.distanceRan));
            results.push(runn.tRex.jumping? 1.0 : 0.0);

            results.push(runn.tRex.xPos);
            results.push((runn.tRex.yPos));
            results.push(runn.currentSpeed);

            if (runn.horizon.obstacles.length > 0) {
                results.push(runn.horizon.obstacles[0].xPos);
                results.push((runn.horizon.obstacles[0].yPos));
            } else {
                results.push(700.0);
                results.push(90.0);
            }
            return results;
        })();"""
    

    INPUT_DIM = 5
    ACTION_DIM = 2

    EPSILON_START = 1.0
    EPSILON_STOP = 0.01
    EPSILON_DECAY = 0.9

    REWARD_DECAY = 0.5

    threshold = {
        "jump":100,
        "duck":75,
        "delta":0
    }

    @classmethod
    def load_brain(cls,model_path):
        return cls.__init__(load_model(model_path))


    def __init__(self,brain=None):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument("--disable-extensions")
        # chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox") # linux only
        # chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(
            './chromedriver',
            options=chrome_options
            )
        self.driver.set_window_rect(
            0,0,
            960,540
        )
        self.driver.get(self.URL)

        time.sleep(0.5)
        self.body = self.driver.find_element_by_tag_name("body")
        time.sleep(1)
        self.move_jump()
        self.epsilon = self.EPSILON_START

        self.memory = deque(maxlen=10_000_000)
        if brain is None:
            self.brain = self.make_brain()
        else:
            self.brain = brain

        self.action_hist = [0,0]
        return

    def __del__(self):
        try:
            self.driver.close()
        except:
            pass # intentional

    def make_brain(self):
        m = Sequential()
        m.add(Dense(
            4,
            activation="relu",
            input_shape=(self.INPUT_DIM,)
        ))
        m.add(Dropout(0.2))
        m.add(Dense(
            self.ACTION_DIM,
            activation='linear'
        ))
        m.compile(
            loss="mse",
            optimizer="adam"
        )
        return m

    
    #### Dino Moves ####

    def move(self,m):
        if m == 0:
            self.move_pass()
        elif m == 1:
            self.move_jump()
        elif m == 2:
            self.move_duck()
        else:
            raise "ERROR! I don't know what move that is: '%s'" % m

    def move_jump(self):
        self.body.send_keys(Keys.UP)

    def move_duck(self):
        self.body.send_keys(Keys.DOWN)

    def move_pass(self):
        pass # Intentional

    def get_positions(self):
        return self.driver.execute_script(self.FN)

    #### RL Functions ####

    def memorize(self,info,state,next,action,reward):
        self.memory.append({
            "info":   info,
            "state":  state,
            "next":   next,
            "action": action,
            "reward": reward
        })

    def get_rewards(self,state):
        return self.brain.predict(state.reshape((1,-1)))[0]

    def get_threshold_action(self,state,timestep):
        if (state[3] < self.threshold['jump'] and 
            state[4] > self.threshold['duck']):
            a = 1
        else:
            a = 0
        self.action_hist[a] += 1
        return a

    def get_action(self,state):
        if np.random.random() < self.epsilon:
            a = np.random.randint(self.ACTION_DIM)
        else:
            rewards = self.get_rewards(state)
            a = np.argmax(rewards)
        self.action_hist[a] += 1
        return a

    def get_minibatch(self, size):
        return np.random.choice(self.memory,size)

    def replay(self,epochs=1,batch_size=128,discount_factor=0.9):
        for e in range(epochs):
            batch = self.get_minibatch(batch_size)

            info   = np.array([b['info'] for b in batch])
            state  = np.array([b['state'] for b in batch])
            next   = np.array([b['next'] for b in batch])
            action = np.array([b['action'] for b in batch]).astype('int')
            reward = np.array([b['reward'] for b in batch])

            predictions = self.brain.predict(state)
            next_preds  = self.brain.predict(next)
            for i, a in enumerate(action):
                predictions[i,a] = reward[i]
                if info[i][0] != 1:
                    predictions[i,a] += discount_factor * np.max(next_preds[i])

            self.brain.fit(
                state,
                predictions,
                batch_size=batch_size,
                epochs=1,
                verbose=0
            )

    def update_epsilon(self):
        if self.epsilon > self.EPSILON_STOP:
            self.epsilon = max(
                self.EPSILON_STOP,
                self.epsilon * self.EPSILON_DECAY
            )

    def calculate_reward(self,full_state):
        reward = 0
        if full_state[0]:
            reward -= 10
        elif full_state[3] > full_state[6]:
            # print("Good job! +5")
            reward += 5
        return reward

    #### Run Training Session ####

    def teach(self,n_episodes=1):
        for episode in range(n_episodes):
            self.move_jump()
            started = False
            time.sleep(0.5)
            done = False
            self.action_hist = [0,0]

            self.threshold = {
                "jump":np.random.normal(140,10),
                "duck":np.random.normal(75,1),
                "delta":np.random.normal(0.01,0.001)
            }

            timestep = 0

            while not done:
                # Extract the positions
                s = self.get_positions()
                done, dist, jumping = s[:3]
                state = np.array(s[3:])
                
                if done and not started:
                    done = False
                    self.move_jump()
                    time.sleep(0.5)
                    continue
                elif not started:
                    started = True

                if jumping and not done:
                    continue
                
                # Choose an action
                action = self.get_threshold_action(state,timestep)

                # Make the move
                self.move(action)

                # calculate reward
                reward = self.calculate_reward(s)

                # Take a lil break
                # time.sleep(0.01) # NOTE: take this out?

                # get next state
                next_state = np.array(self.get_positions())[3:]
                    
                # store that information
                self.memorize(
                    [done,dist,jumping],
                    state,
                    next_state,
                    action,
                    reward
                    )

                timestep += 1

            print(f"EPISODE: {episode:3d} | DISTANCE RAN: {dist:10.2f} | JUMP THRESHOLD: {self.threshold['jump']:.2f} | DUCK THRESHOLD: {self.threshold['duck']:.2f}")

            self.update_epsilon()
            self.replay(5)

        return

    def train(self,n_episodes=5):
        final_dists = []
        self.epsilon = self.EPSILON_START
        for episode in range(n_episodes):
            self.move_jump()
            started = False
            time.sleep(0.5)
            done = False
            self.action_hist = [0,0]

            while not done:
                # Extract the positions
                s = self.get_positions()
                done, dist, jumping = s[:3]
                state = np.array(s[3:])
                
                if done and not started:
                    done = False
                    self.move_jump()
                    time.sleep(0.5)
                    continue
                elif not started:
                    started = True

                # if jumping and not done:
                #     continue
                
                # Choose an action
                action = self.get_action(state)

                # Make the move
                self.move(action)

                # calculate reward
                reward = self.calculate_reward(s)

                # Take a lil break
                time.sleep(0.01) # NOTE: take this out?

                # get next state
                next_state = np.array(self.get_positions())[3:]
                    
                # store that information
                self.memorize(
                    [done,dist,jumping],
                    state,
                    next_state,
                    action,
                    reward
                    )

            final_dists.append(dist)

            total = sum(self.action_hist)
            if total:
                actions = [round(a/total,4) for a in self.action_hist]
            else:
                actions = self.action_hist
            print(f"EPISODE: {episode:3d} | DISTANCE RAN: {dist:10.2f} | EPSILON: {self.epsilon:.4f} | ACTIONS: {actions}")

            self.update_epsilon()
            self.replay(5)

        return final_dists

    
    def save_brain(self,filepath):
        save_model(
            self.brain,
            filepath
            )



    
if __name__ == "__main__":
    dt_fmt = time.strftime("%Y%m%d.%H%M%S")
    print("Program starting")
    print("building dino...")
    try:
        runner = DinoGame()
        print("Starting game")
        print("Teaching by example...")
        runner.teach(5)
        print("Training RL...")
        dists = runner.train(1000)
    finally:
        runner.driver.close()
        print("Done")

    runner.save_brain(f"models/brain_{dt_fmt}.h5")

    plt.plot(dists)
    plt.title("tRex Distance Traveled")
    plt.savefig(f"./distplots/trex_dist_plot{dt_fmt}.png")




