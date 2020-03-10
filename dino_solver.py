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
            for (let i = 0; i < 3; i++) {
                if (i < runn.horizon.obstacles.length) {
                    results.push(
                        runn.horizon.obstacles[i].xPos
                    );
                    results.push(
                        runn.horizon.obstacles[i].yPos
                    );
                } else {
                    results.push(500);
                    results.push(90);
                }
            }
            return results;
        })();""" 

    # output from js: (
    #     crashed?,
    #     distRan,
    #     tRex.X
    #     tRex.Y,
    #     obs1.X,
    #     obs1.Y,
    #     obs2.X,
    #     obs2.Y,
    #     obs3.X,
    #     obs3.Y
    # )

    INPUT_DIM = 8
    ACTION_DIM = 3

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
        pred = self.brain.predict(state.reshape((1,-1)))[0]
        a = np.argmax(pred)
        self.action_hist[a] += 1
        return a

    def replay(self,epochs=1,batch_size=32):
        for e in range(epochs):
            self.move_jump()
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
                    obs2X,
                    obs2Y,
                    obs3X,
                    obs3Y,
                    action,
                    reward
                ) = s
                # Get the brain's current prediction
                model_in = np.array([
                    tRexX,
                    tRexY,
                    obs1X,
                    obs1Y,
                    obs2X,
                    obs2Y,
                    obs3X,
                    obs3Y
                ])
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
                pred[action] = reward

                # Retrain
                self.brain.fit(
                    model_in,
                    [pred],
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
            while not done:
                # Extract the positions
                s = self.get_positions()
                (
                    done,
                    dist,
                    tRexX,
                    tRexY,
                    obs1X,
                    obs1Y,
                    obs2X,
                    obs2Y,
                    obs3X,
                    obs3Y
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
                    obs1Y,
                    obs2X,
                    obs2Y,
                    obs3X,
                    obs3Y
                ))
                # print("State shape:",state.shape)
                action = self.get_action(state)

                # Make the move
                self.move(action)

                # calculate reward
                reward = 1

                # store that information
                self.memorize(s,action,reward)

                # Take a lil break
                time.sleep(0.1) # NOTE: adjust this time?
            final_dists.append(dist)
            
            CRASH_BACWARD_DEPTH = 30
            CRASH_REWARD_DECAY = 0.85
            CRASH_REWARD = -5
            REWARD_MEMORY_POS = -1
            for i in range(CRASH_BACWARD_DEPTH):
                self.memory[-(1+i)][REWARD_MEMORY_POS] += (
                    CRASH_REWARD * CRASH_REWARD_DECAY ** i
                )
            print(f"EPISODE: {episode:3d} | DISTANCE RAN: {dist:20.2f}")
        return final_dists

    
    def save_brain(self,filepath):
        save_model(
            self.brain,
            filepath
            )


    def play(self):
        raise NotImplementedError



    
if __name__ == "__main__":
    dt_fmt = time.strftime("%Y%m%d.%H%M%S")
    print("Program starting")
    print("building dino...")
    try:
        runner = DinoGame()
        print("Starting game")
        dists = runner.train(20)
    finally:
        runner.driver.close()
        print("Done")

    df = pd.DataFrame(
        runner.memory,
        columns=[
            "done",
            "dist",
            "tRexX",
            "tRexY",
            "obs1X",
            "obs1Y",
            "obs2X",
            "obs2Y",
            "obs3X",
            "obs3Y",
            "action",
            "reward"
        ]
    )

    runner.save_brain(f"models/brain_{dt_fmt}.h5")

    with open(f"logs/log_{dt_fmt}.txt",'w') as f:
        f.write(f"Best dist: {df.dist.max()}\n")
        f.write(f"Avg reward: {df.reward.mean()}\n")
        f.write("\n\nAction breakdown:\n")
        f.write(str(df.action.value_counts()))

    plt.plot(dists)
    plt.title("tRex Distance Traveled")
    plt.savefig(f"./images/trex_dist_plot{dt_fmt}.png")




