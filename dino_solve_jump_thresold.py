"""
dino_solver.py

by Austin Poor


Plays the chrome-dino-game based on rules
specified beforehand.

Also used to find the proper thresholds to use
for deciding actions using grid-search.

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
            results.push(runn.distanceMeter.getActualDistance(runn.distanceRan));
            results.push(runn.tRex.jumping? 1.0 : 0.0);
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



    def __init__(self, jt_vals, dt_vals, jt_deltas, n_tests=10):
        self.jt_vals = jt_vals
        self.dt_vals = dt_vals
        self.jt_deltas = jt_deltas
        self.n_tests = n_tests
        self.threshold = None
        self.thresh_data = {}

        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        
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
        self.driver.execute_script("(new Runner()).playCount = 100;")
        
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
                return 0
            else:
                return 1
        else:
            return 0

    #### Run Training Session ####

    def train(self,n_episodes=5):
        final_dists = []

        total_tests = len(self.jt_vals) * len(self.dt_vals) * len(self.jt_deltas) * self.n_tests
        current_test = 0

        dist_hist = []

        for jth in self.jt_vals:
            for dth in self.dt_vals:
                for jtd in self.jt_deltas:
                    print(f"Testing jump thresold = {jth:.2f} and duck threshold = {dth:.2f} and jump threshold delta = {jtd} for {self.n_tests} tests")
                    for episode in range(self.n_tests):
                        dist_hist.append([])
                        self.move_jump()
                        started = False
                        time.sleep(1)
                        done = False

                        last_state = np.array(self.get_positions()[3:]) #NOTE: get the last position
                        time_step = 0

                        jt_delta_acc = 0

                        while not done:
                            # Extract the positions
                            s = self.get_positions()
                            done, dist, jumping = s[:3]
                            speed = s[5]

                            dist_hist[-1].append((
                                dist,
                                speed
                            ))
                            
                            if done and not started:
                                done = False
                                self.move_jump()
                                time.sleep(0.5)
                                continue
                            elif not started:
                                started = True


                            if jumping:
                                continue
                            
                            # Choose an action
                            state = np.array(s[3:])

                            # print("State shape:",state.shape)
                            action = self.get_action(
                                state,
                                last_state,
                                jth + jt_delta_acc,
                                dth
                                )

                            if speed < 12:
                                jt_delta_acc += jtd

                            # Make the move
                            self.move(action)

                            # pass back the last state
                            last_state = state

                            # Take a lil break
                            # time.sleep(0.01) #NOTE: Taking out the break for now

                            time_step += 1
                        
                        current_test += 1
                        final_dists.append(dist)
                        params = (jth,dth,jtd)
                        self.thresh_data[params] = self.thresh_data.get(params,[]) + [dist]
                        print(f"EPISODE: {episode:3d} ({current_test:5d}/{total_tests}) | DISTANCE RAN: {dist:10.2f} | FINAL SPEED: {speed} | JUMP THRESOLD: {jth:.2f} | TIME STEPS: {time_step}")
        # for dh in dist_hist:
        #     dx, dy = zip(*dh)
        #     sns.scatterplot(dx,dy)
        # plt.show()
        # input()
        return final_dists


    
if __name__ == "__main__":
    print("Program starting")
    print("building dino...")
    try:
        runner = DinoGame(
            jt_vals=np.linspace(50,150,10).round(2),
            dt_vals=[75],
            jt_deltas=[.01], #np.linspace(0.1,0.05,5).round(2),
            n_tests=3
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
            len(runner.jt_deltas)
        )),
        columns=runner.jt_deltas,
        index=runner.jt_vals
    )
    for (jt,dt,jd), vals in runner.thresh_data.items():
        df.loc[jt,jd] = np.mean(vals)
    sns.heatmap(df)
    plt.title("Chrome Dino Game\nAverage Distance Hyperperameter Grid Search")
    plt.xlabel("Jump Delta")
    plt.ylabel("Jump Threshold")
    plt.show()


