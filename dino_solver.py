"""
dino_solver.py

by Austin Poor



NOTES:
– Run for longer to see how it changes
– save the memory to a SQLite DB so that it doesn't
  need to re-learn everything every time
– give it a past frame in addition to current (or more)
– can you scale the inputs? (how would you handle)
– use more/less layers? to improve training?
– add in random actions (epsilon-greedy)


– run multiple training sessions simultaneously
– run training headless in the cloud?



### This Version:
– Limiting the number of enemies seen
– Updating rewards
  * zero reward for pass
  * positive reward for jump
– Creates queue for a single episode where reward can
  be propagated backward before being added.

"""



import time
from collections import deque

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

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
            results.push(runn.crashed? 1 : 0);
            results.push(runn.distanceRan);
            results.push(runn.tRex.xPos);
            results.push(runn.tRex.yPos);
            if (runn.horizon.obstacles.length > 0) {
                results.push(runn.horizon.obstacles[0].xPos);
                results.push(runn.horizon.obstacles[0].yPos);
            } else {
                results.push(500);
                results.push(90);
            }
            return results;
        })();""" 

    # output from js: (
    #     crashed?,
    #     distRan,
    #     tRex.X
    #     tRex.Y,
    #     obs.X,
    #     obs.Y,
    # )

    INPUT_DIM = 4
    ACTION_DIM = 3

    EPSILON_START = 1.0
    EPSILON_STOP = 0.01
    EPSILON_DECAY = 0.995

    @classmethod
    def load_brain(cls,model_path):
        return cls.__init__(load_model(model_path))


    def __init__(self,brain=None):
        self.driver = webdriver.Chrome('./chromedriver')
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

        self.action_hist = [0,0,0]
        return

    def __del__(self):
        try:
            self.driver.close()
        except:
            pass # intentional

    def make_brain(self):
        m = Sequential()
        m.add(Dense(
            16,
            activation="relu",
            input_shape=(self.INPUT_DIM,)
        ))
        m.add(Dropout(0.2))
        m.add(Dense(
            16,
            activation="relu"
        ))
        m.add(Dropout(0.2))
        # m.add(Dense(
        #     16,
        #     activation="relu"
        # ))
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

    def memorize(self,state_info,action,reward=1):
        self.memory.append(np.concatenate((
            state_info,
            (
                action,
                reward
            )
        )))
        return

    def get_action(self,state):
        if np.random.random() < self.epsilon:
            a = np.random.randint(3)
        else:
            pred = self.brain.predict(state.reshape((1,-1)))[0]
            a = np.argmax(pred)
        if self.epsilon > self.EPSILON_STOP:
            self.epsilon = max(
                self.EPSILON_STOP,
                self.epsilon * self.EPSILON_DECAY
            )
        self.action_hist[a] += 1
        return a

    def replay(self,epochs=1,batch_size=32):
        for e in range(epochs):
            time.sleep(1)
            for b in range(batch_size):
                # Pick a random memory state
                i = np.random.randint(len(self.memory))
                s = self.memory[i]
                # Unpack the data
                (
                    _,
                    _,
                    tRexX,
                    tRexY,
                    obs1X,
                    obs1Y,
                    action,
                    reward
                ) = s
                # Get the brain's current prediction
                model_in = np.array([[
                    tRexX,
                    tRexY,
                    obs1X,
                    obs1Y
                ]])
                pred = self.brain.predict(model_in)[0]

                # NOTE: Add extra reward if not done
                # and obstacle behind player
                # reward = 1
                # if crashed:
                #     reward = -5
                # else: 
                #     # NOTE: This is where we'll add
                #     # the predicted future reward
                #     reward += 0.9
                
                # Update the reward
                pred[int(action)] = reward
                pred = pred.reshape((1,-1))

                # Retrain
                self.brain.fit(
                    model_in,
                    pred,
                    epochs=1,
                    verbose=0
                )

    #### Run Training Session ####

    def train(self,n_episodes=5):
        final_dists = []
        for episode in range(n_episodes):
            self.move_jump()
            started = False
            time.sleep(1)
            done = False
            mem_buff = []
            while not done:
                # Extract the positions
                s = self.get_positions()
                (
                    done,
                    dist,
                    tRexX,
                    tRexY,
                    obs1X,
                    obs1Y
                ) = s
                if done and not started:
                    done = False
                    self.move_jump()
                    time.sleep(0.5)
                    continue
                elif not started:
                    started = True
                
                # Choose an action
                state = np.array((
                    tRexX,
                    tRexY,
                    obs1X,
                    obs1Y
                ))
                # print("State shape:",state.shape)
                action = self.get_action(state)

                # Make the move
                self.move(action)

                # calculate reward
                if done:
                    reward = -10
                else:
                    reward = 0
                    if action != 0:
                        reward += 0 #-0.1
                    if tRexX > obs1X:
                        print("Good job! +5")
                        reward += 5
                    

                # store that information
                mem_buff.append([s,action,reward])

                # Take a lil break
                time.sleep(0.1) # NOTE: adjust this time?
            final_dists.append(dist)
            
            
            # Update the past reward in mem_buff
            REWARD_DECAY = 0.9
            for i in range(len(mem_buff)-2,-1,-1):
                mem_buff[i][2] += mem_buff[i+1][2] * REWARD_DECAY

            # add those to the full memory
            for s, action, reward in mem_buff:
                self.memorize(s,action,reward)

            self.replay(5)


            print(f"EPISODE: {episode:3d} | DISTANCE RAN: {dist:10.2f} | AVG REWARD: {sum(s[2] for s in mem_buff)/len(mem_buff):5.2f}")
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
        dists = runner.train(500)
    finally:
        runner.driver.close()
        print("Done")

    runner.save_brain(f"models/brain_{dt_fmt}.h5")

    plt.plot(dists)
    plt.title("tRex Distance Traveled")
    plt.savefig(f"./images/trex_dist_plot{dt_fmt}.png")




