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

    # (
    #     crashed,
    #     distance,

    #     tRexX,
    #     tRexY,

    #     speed,

    #     o1X,
    #     o1Y,
    #     o1W,
    #     o1H,

    #     o2X,
    #     o2Y,
    #     o2W,
    #     o2H,

    #     o3X,
    #     o3Y,
    #     o3W,
    #     o3H
    # )


    def __init__(self, jt_vals, dt_vals, n_tests=10):
        self.jt_vals = jt_vals
        self.dt_vals = dt_vals
        self.n_tests = n_tests
        self.threshold = None
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
        self.body.send_keys(Keys.SPACE)

    def move_duck(self):
        self.body.send_keys(Keys.DOWN)

    def move_pass(self):
        pass # Intentional

    def get_positions(self):
        return self.driver.execute_script(self.FN)

    #### RL Functions ####

    def get_action(self,current,last,jump_threshold,duck_threshold):
        if current[3] < jump_threshold:
            if current[4] < duck_threshold:
                # print("Ducking...")
                return 2
            else:
                # print("Jumping...")
                return 1
        else:
            return 0

    #### Run Training Session ####

    def train(self,n_episodes=5):
        final_dists = []

        total_tests = len(self.jt_vals) * len(self.dt_vals) * self.n_tests
        current_test = 0
        for jth in self.jt_vals:
            for dth in self.dt_vals:
                print(f"Testing jump thresold = {jth:.2f} and duck threshold = {dth:.2f} for {self.n_tests} tests")
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
                        action = self.get_action(state,last_state,jth,dth)

                        # Make the move
                        self.move(action)

                        # pass back the last state
                        last_state = state

                        # Take a lil break
                        time.sleep(0.1)
                    
                    current_test += 1
                    final_dists.append(dist)
                    params = (jth,dth)
                    self.thresh_data[params] = self.thresh_data.get(params,[]) + [dist]
                    print(f"EPISODE: {episode:3d} ({current_test:5d}/{total_tests}) | DISTANCE RAN: {dist:10.2f} | JUMP THRESOLD: {jth:.2f} | DUCK THRESOLD: {dth:.2f}")
        return final_dists


    
if __name__ == "__main__":
    print("Program starting")
    print("building dino...")
    try:
        runner = DinoGame(
            jt_vals=np.linspace(150,190,5),
            dt_vals=np.linspace(60,80,5),
            n_tests=5
            )
        print("Starting game")
        dists = runner.train(500)
    finally:
        runner.driver.close()
        print("Done")

    # sns.scatterplot(*zip(*[(k,sum(v)/len(v)) for k, v in runner.thresh_data.items()]))
    df = pd.DataFrame(
        np.zeros((
            len(runner.jt_vals),
            len(runner.dt_vals)
        )),
        columns=runner.dt_vals,
        index=runner.jt_vals
    )
    for (j,d), vals in runner.thresh_data.items():
        df.loc[j,d] = np.mean(vals)
    sns.heatmap(df)
    plt.title("Chrome Dino Game\nAverage Distance Hyperperameter Grid Search")
    plt.xlabel("Duck Threshold (\"Vertical\")")
    plt.ylabel("Jump Threshold (\"Horizontal\")")
    plt.show()


