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

REWARD_DECAY = 0.9

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

    columns = [
        "crashed",
        "distance",

        "tRexX",
        "tRexY",

        "speed",

        "o1X",
        "o1Y",
        "o1W",
        "o1H",

        "o2X",
        "o2Y",
        "o2W",
        "o2H",

        "o3X",
        "o3Y",
        "o3W",
        "o3H"
    ]
    

    INPUT_DIM = 15
    ACTION_DIM = 3

    EPSILON_START = 1.0
    EPSILON_STOP = 0.01
    EPSILON_DECAY = 0.9995

    REWARD_DECAY = 0.5

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
    def get_rewards(self,hist,acts):
        rewards = []
        last_r = 0
        for s, a in zip(hist[::-1],acts[::-1]):
            if s[0]:
                r = -10
            elif s[3] < 0:
                r = 5
            else:
                r = 0
            if a > 0:
                r += 0 #-0.5
            r += last_r * REWARD_DECAY
            rewards.append(r)
            last_r = r
        return rewards[::-1]

    def store_history(self,hist,actions,rewards):
        c = self.db.cursor()
        for s, a, r in zip(hist,actions,rewards):
            c.execute("""INSERT INTO "DinoGame" (
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
                    o3H,
                    action,
                    reward
                ) VALUES (
                    ?,?,?,?,?,
                    ?,?,?,?,?,
                    ?,?,?,?,?,
                    ?,?,?,?
                )""",
                s + [a,r]
                )
        self.db.commit()

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

    def replay(self,epochs=1,batch_size=256):
        c = self.db.cursor()

        c.execute("SELECT COUNT(*) FROM DinoGame;")
        if c.fetchone()[0] < batch_size:
            return
        for _ in range(epochs):
            c.execute("""
                SELECT 
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
                    o3H,
                    action,
                    reward 
                FROM 
                    DinoGame 
                ORDER BY 
                    Random() 
                LIMIT ?;""",(batch_size,))
            

            def decode_number(n):
                try:
                    return float(n)
                except ValueError:
                    return float(int.from_bytes(n,'little'))

            data = [[decode_number(n) for n in l] for l in c.fetchall()]
            data = np.array(data)

            actions, rewards = data[:,-2], data[:,-1]

            try:
                model_in = data[:,2:-2]
                predictions = self.brain.predict(model_in)
            except:
                print(type(model_in))
                print(model_in)
                raise


            predictions[:,actions.astype('int32')] = rewards

            # Retrain
            self.brain.fit(
                model_in,
                predictions,
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

            state_history = []
            action_history = []

            while not done:
                # Extract the positions
                s = self.get_positions()
                done, dist = s[:2]
                
                if done and not started:
                    done = False
                    self.move_jump()
                    time.sleep(0.5)
                    continue
                elif not started:
                    started = True
                
                # start_time = time.perf_counter()
                # Choose an action
                state = np.array(s[2:])

                # Get the action
                action = self.get_action(state)

                # Make the move
                self.move(action)

                # store that information
                state_history.append(s)
                action_history.append(action)

                # Take a lil break
                # time_delta = time.perf_counter() - start_time
                # pause_time = 0.1 #0.01
                # time.sleep(max(0,pause_time-time_delta)) # NOTE: adjust this time?

                time.sleep(0.1)

            final_dists.append(dist)

            rewards = self.get_rewards(state_history,action_history)
            self.store_history(state_history,action_history,rewards)
            self.replay(32)

            print(f"EPISODE: {episode:3d} | DISTANCE RAN: {dist:10.2f} | AVG REWARD: {sum(rewards)/len(rewards)}")
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




