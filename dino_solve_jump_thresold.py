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
import seaborn as sns
sns.set()

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


    def __init__(self,jt_min=-50,jt_max=600,n_steps=20,n_tests=10):
        self.jt_min = jt_min
        self.jt_max = jt_max
        self.n_steps = n_steps
        self.n_tests = n_tests
        self.threshold = self.jt_max
        self.thresh_data = {}

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

        self.memory = deque(maxlen=10_000_000)
        return

    def __del__(self):
        try:
            self.driver.close()
        except:
            pass # intentional

    
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

    def get_action(self,current,last):
        if (current[3] <= self.threshold): # and (last[3] > self.threshold):
            return 1
        else:
            return 0

    #### Run Training Session ####

    def train(self,n_episodes=5):
        final_dists = []
        self.epsilon = self.EPSILON_START # NOTE: restart every episode?
        thresolds = np.linspace(
            self.jt_min,
            self.jt_max,
            self.n_steps
            )
        for th in thresolds:
            self.threshold = th
            print("Testing thresold: ", th, "  , ", self.n_tests, " times")
            for episode in range(self.n_tests):
                self.move_jump()
                started = False
                time.sleep(1)
                done = False

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

                    # print("State shape:",state.shape)
                    action = self.get_action(state,last_state)

                    # Make the move
                    self.move(action)

                    # pass back the last state
                    last_state = state

                    # Take a lil break
                    time.sleep(0.1) # NOTE: adjust this time?

                final_dists.append(dist)
                self.thresh_data[self.threshold] = self.thresh_data.get(self.threshold,[]) + [dist]
                print(f"EPISODE: {episode:3d} | DISTANCE RAN: {dist:10.2f} | THRESOLD: {th}")
        return final_dists



    
if __name__ == "__main__":
    print("Program starting")
    print("building dino...")
    try:
        runner = DinoGame(jt_min=0,jt_max=400,n_steps=150,n_tests=10)
        print("Starting game")
        dists = runner.train(500)
    finally:
        runner.driver.close()
        print("Done")

    sns.scatterplot(*zip(*[(k,sum(v)/len(v)) for k, v in runner.thresh_data.items()]))                                                
    plt.show()
    input("Press enter to quit")
    plt.close()


