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
import sqlite3

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
            results.push(runn.crashed? 1 : 0);
            results.push(runn.distanceMeter.getActualDistance(runn.distanceRan));
            results.push(runn.tRex.xPos);
            results.push((runn.tRex.yPos));
            results.push(runn.currentSpeed);
            for (let i = 0; i < 3; i++) {
                if (runn.horizon.obstacles.length > i) {
                    results.push(runn.horizon.obstacles[i].xPos);
                    results.push((runn.horizon.obstacles[i].yPos));
                    results.push(runn.horizon.obstacles[i].typeConfig.width);
                    results.push(runn.horizon.obstacles[i].typeConfig.height);
                } else {
                    results.push(650);
                    results.push(90);
                    results.push(0);
                    results.push(0);
                }
            }
            return results;
        })();"""

    _ = """(
            crashed,
            distance,

            tRexX,
            tRexY,

            speed,

            o1X,
            o1Y,
            o1W,
            o1H,

            o2X,
            o2Y,
            o2W,
            o2H,

            o3X,
            o3Y,
            o3W,
            o3H
        )"""
    

    INPUT_DIM = 30
    ACTION_DIM = 3

    EPSILON_START = 1.0
    EPSILON_STOP = 0.01
    EPSILON_DECAY = 0.9995

    REWARD_DECAY = 0.5

    # DB_PATH = "dino_memory2.db"
    DB_PATH = "dino_mem.db"

    @classmethod
    def load_brain(cls,model_path):
        return cls.__init__(load_model(model_path))


    def __init__(self,brain=None):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
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

        self.db = sqlite3.connect(self.DB_PATH)
        c = self.db.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS Memory (
            crashed REAL,
            distance REAL,

            tRexX REAL,
            tRexY REAL,

            speed REAL,

            obs1X REAL,
            obs1Y REAL,
            obs1W REAL,
            obs1H REAL,

            obs2X REAL,
            obs2Y REAL,
            obs2W REAL,
            obs2H REAL,

            obs3X REAL,
            obs3Y REAL,
            obs3W REAL,
            obs3H REAL,

            last_tRexX REAL,
            last_tRexY REAL,

            last_speed REAL,

            last_obs1X REAL,
            last_obs1Y REAL,
            last_obs1W REAL,
            last_obs1H REAL,

            last_obs2X REAL,
            last_obs2Y REAL,
            last_obs2W REAL,
            last_obs2H REAL,

            last_obs3X REAL,
            last_obs3Y REAL,
            last_obs3W REAL,
            last_obs3H REAL,

            action REAL,
            reward REAL
        );""")

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
            32,
            activation="relu",
            input_shape=(self.INPUT_DIM,)
        ))
        m.add(Dropout(0.2))
        m.add(Dense(
            32,
            activation="relu"
        ))
        m.add(Dropout(0.2))
        m.add(Dense(
            32,
            activation="relu"
        ))
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

    def memorize(self,state_info,last_state_info,action,reward):
        d = np.concatenate((
            state_info,
            last_state_info,
            (
                action,
                reward
            )
        ))
        c = self.db.cursor()
        c.execute("""INSERT INTO Memory (
            crashed,
            distance,

            tRexX,
            tRexY,

            speed,

            obs1X,
            obs1Y,
            obs1W,
            obs1H,

            obs2X,
            obs2Y,
            obs2W,
            obs2H,

            obs3X,
            obs3Y,
            obs3W,
            obs3H,

            last_tRexX,
            last_tRexY,

            last_speed,

            last_obs1X,
            last_obs1Y,
            last_obs1W,
            last_obs1H,

            last_obs2X,
            last_obs2Y,
            last_obs2W,
            last_obs2H,

            last_obs3X,
            last_obs3Y,
            last_obs3W,
            last_obs3H,

            action,
            reward
        ) VALUES (
            ?,?,?,?,?,
            ?,?,?,?,?,
            ?,?,?,?,?,
            ?,?,?,?,?,
            ?,?,?,?,?,
            ?,?,?,?,?,
            ?,?,?,?
        );""",d)
        self.db.commit()
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

    def replay(self,epochs=1,batch_size=128):
        c = self.db.cursor()
        for e in range(epochs):
            time.sleep(1)
            data_in = []
            data_out = []
            for b in range(batch_size):
                # Pick a random memory state
                c.execute("""SELECT * FROM Memory ORDER BY Random() LIMIT 1;""")
                s = np.array(c.fetchone())

                # Unpack the data
                action, reward = s[-2:]

                # Get the brain's current prediction
                model_in = np.array(s[2:-2])
                pred = self.brain.predict(model_in.reshape((1,-1)))[0]
                
                # Update the reward
                pred[int(action)] = reward
                # pred = pred.reshape((1,-1))

                data_in.append(model_in)
                data_out.append(pred)

            # Retrain
            self.brain.fit(
                np.array(data_in),
                np.array(data_out),
                epochs=1,
                verbose=0
            )

    #### Run Training Session ####

    def train(self,n_episodes=5):
        final_dists = []
        self.epsilon = self.EPSILON_START # NOTE: restart every episode?
        for episode in range(n_episodes):
            self.move_jump()
            started = False
            time.sleep(1)
            done = False
            mem_buff = []

            last_state = np.array(self.get_positions()[2:]) #NOTE: get the last position
            while not done:
                # Extract the positions
                s = self.get_positions()
                done, dist = s[:2]
                tRexX = s[2]
                obs1X = s[5]
                
                if done and not started:
                    done = False
                    self.move_jump()
                    time.sleep(0.5)
                    continue
                elif not started:
                    started = True
                
                # Choose an action
                state = np.array(s[2:])
                full_state = np.concatenate((
                    state,
                    last_state
                ))
                # print("State shape:",state.shape)
                action = self.get_action(full_state)

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
                        # print(s)
                    

                # store that information
                mem_buff.append([s,last_state,action,reward])

                # pass back the last state
                last_state = state

                # Take a lil break
                time.sleep(0.1) # NOTE: adjust this time?
            final_dists.append(dist)
            
            
            # Update the past reward in mem_buff
            for i in range(len(mem_buff)-2,-1,-1):
                mem_buff[i][3] += mem_buff[i+1][3] * self.REWARD_DECAY

            # add those to the full memory
            for s, ls, action, reward in mem_buff:
                self.memorize(s,ls,action,reward)

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




