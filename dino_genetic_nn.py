"""
dino_genetic_nn.py

by Austin Poor



NOTES:
– run multiple training sessions simultaneously
– run training headless in the cloud?

"""



import os
import time
from copy import deepcopy

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model, save_model, clone_model
from tensorflow.keras.layers import Dense, Dropout


class Game:

    URL = 'chrome://dino/'
    CSS_RUNNER_CONTAINER = '.runner-container'

    FN = """return (function () {
            const results = [];
            const runn = new Runner();
            results.push(runn.tRex.jumping? 1.0 : 0.0);
            results.push(runn.crashed? 1 : 0);
            results.push(runn.distanceMeter.getActualDistance(runn.distanceRan));
            results.push(runn.tRex.xPos / 700.0);
            results.push(runn.tRex.yPos /  90.0);
            results.push(runn.currentSpeed);
            if (runn.horizon.obstacles.length > 0) {
                results.push(runn.horizon.obstacles[0].xPos / 700.0);
                results.push(runn.horizon.obstacles[0].yPos /  90.0);
            } else {
                results.push(1.0);
                results.push(1.0);
            }
            return results;
        })();"""


    def __init__(self):
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
        # self.move_jump()

    def __del__(self):
        try:
            self.driver.close()
        except:
            pass # intentional

    def restart(self):
        self.driver.get(self.URL)
        time.sleep(0.5)
        self.body = self.driver.find_element_by_tag_name("body")

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

    def score_brain(self,B,n_games=5,prefix=''):
        dists = []
        for _ in range(n_games):
            self.restart()
            done = False
            self.move_jump()
            time.sleep(0.5)

            while not done:
                s = self.get_positions()
                jumping, done, dist = s[:3]
                s = s[1:]
                
                if jumping:
                    continue

                # Choose an action
                state = np.array(s[2:])

                # Get the action
                action = B.get_action(state)

                # Make the move
                self.move(action)

                # Sleep for a (fraction of a) sec
                time.sleep(0.01)

            dists.append(dist)
        avg_dist = np.mean(dists)
        print(f"{prefix:5s} | n_games: {n_games} | avg_dist: {avg_dist:.4f}")
        return avg_dist




class Population:
    def __init__(self, pop_size, crossover_rate, mutation_rate, n_subpop, n_games):
        self.pop_size = pop_size
        self.xover = crossover_rate
        self.mrate = mutation_rate
        self.n_subpop = n_subpop
        self.n_games = n_games
        self.fitness_history = []
        self.timestep = 0
        
        self.game = Game()

        self.initialize_population()

        return

    def __del__(self):
        try:
            self.game.driver.close()
        except:
            pass

    def initialize_population(self):
        self.population = [DinoBrain() for _ in range(self.pop_size)]
        self.fitnesses = [None for _ in range(self.pop_size)]

    def get_fitnesses(self):
        self.fitnesses = [
            self.game.score_brain(b,self.n_games,hex(i)) for i, b in enumerate(self.population)
        ]
        stats = {
            "min": round(min(self.fitnesses),4),
            "mean": round(sum(self.fitnesses) / len(self.fitnesses),4),
            "max": round(max(self.fitnesses),4)
        }
        self.fitness_history.append(stats)
        return stats

    def step(self):
        print(f" –– TIMESTEP: {self.timestep} ––")
        stats = self.get_fitnesses()
        print(f"POP FITNESSES:{stats}")

        new_pop = []
        best_brain = self.population[np.argmax(self.fitnesses)]
        new_pop.append(best_brain)

        # probs = np.argsort(self.fitnesses).astype('float32')
        # probs = sum(probs) - probs
        # probs = probs / sum(probs)

        subpop_i = np.argsort(self.fitnesses)[::-1][:min(self.pop_size,self.n_subpop)]
        subpop_b = [self.population[i] for i in subpop_i]
        subpop_f = [self.fitnesses[i] for i in subpop_i]

        ft = sum(subpop_f)
        probs = [f/ft for f in subpop_f]

        print("PERFORMING SELECTION...")
        while len(new_pop) < self.pop_size:
            s1, s2 = self.selection(
                subpop_b,
                probs,
                2
            )
            s1, s2 = s1.copy(), s2.copy()
            self.crossover(s1, s2)
            self.mutate(s1)
            self.mutate(s2)
            new_pop.extend([s1,s2])
        self.population = new_pop
        print()
        self.timestep += 1

    def run(self, n_steps=1, save=True):
        for _ in range(n_steps):
            self.step()
        self.get_fitnesses()
        self.save_brains()

    def plot_hist(self):
        pass

    def save_brains(self):
        dt_fmt = time.strftime("%Y%m%d.%H%M%S")
        dirpath = os.path.join("ga_nn_models",dt_fmt)
        os.mkdir(dirpath)
        for b, f in zip(self.population, self.fitnesses):
            b.save(dirpath, f)

    def selection(self, pop, probs, n):
        return [g.copy() for g in np.random.choice(pop,size=n,p=probs)]

    def crossover(self, a, b):
        aw = a.get_weights()
        bw = b.get_weights()
        for i in range(len(aw)):
            if len(aw[i].shape) == 1:
                for j in range(aw[i].shape[0]):
                    if np.random.random() < self.xover:
                        aw[i][j], bw[i][j] = bw[i][j], aw[i][j]
            else:
                for j in range(aw[i].shape[0]):
                    for k in range(aw[i][j].shape[0]):
                        if np.random.random() < self.xover:
                            aw[i][j][k], bw[i][j][k] = bw[i][j][k], aw[i][j][k]
        a.set_weights(aw)
        a.set_weights(bw)

    def mutate(self, g):
        g.mutate(self.mrate)


class DinoBrain:
    INPUT_DIM = 5
    THRESHOLD = 0.5

    @classmethod
    def from_weights(cls,w):
        return DinoBrain().set_weights(w)

    @classmethod
    def from_h5(cls,f):
        return DinoBrain(load_model(f))
    

    def __init__(self, brain=None):
        if brain is None:
            self.brain = self.make_brain()
        else:
            self.brain = brain
        return

    def make_brain(self):
        m = Sequential()
        m.add(Dense(4,
            activation="relu",
            input_shape=(self.INPUT_DIM,)))
        m.add(Dense(1,
            activation='sigmoid'))
        m.compile(
            loss="binary_crossentropy",
            optimizer="adam")
        return m

    def get_action(self,state):
        r = self.brain.predict(state.reshape((1,-1)))
        return r[0] > self.THRESHOLD

    def copy(self):
        return DinoBrain(clone_model(self.brain))

    def get_weights(self):
        return deepcopy(self.brain.get_weights())

    def set_weights(self,w):
        self.brain.set_weights(w)
        return self

    def mutate(self,mutation_rate,mean=0,std=1.0):
        weights = self.brain.get_weights()
        for i, w in enumerate(weights):
            s = w.shape
            r = np.random.random(s) > mutation_rate
            e = np.random.normal(mean,std,s)
            n = r * e
            weights[i] = w + n
        self.brain.set_weights(weights)

    def save(self, directory, fitness):
        filename = os.path.join(directory, f"ga_dino_model_{fitness:.2f}fit.h5")
        save_model(self.brain, filename)

        

    
if __name__ == "__main__":
    dt_fmt = time.strftime("%Y%m%d.%H%M%S")
    pop = Population(
        40,
        0.9,
        0.01,
        10,
        3
    )
    pop.run(10)




