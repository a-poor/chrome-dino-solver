
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from collections import deque

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.

URL = 'chrome://dino/'
CSS_RUNNER_CONTAINER = '.runner-container'

driver = webdriver.Chrome('./chromedriver')
driver.get(URL)


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

    def __init__(self,brain=None):
        self.driver = webdriver.Chrome('./chromedriver')
        driver.get(self.URL)
        self.body = self.driver.find_element_by_tag_name("body")

        self.memory = deque(maxlen=10_000)
        if brain is None:
            self.brain = self.make_brain()
        else:
            self.brain = brain
        return

    def __del__(self):
        self.driver.close()

    def make_brain(self):
        m = Sequential()
        m.add(Dense())
        return

    
    #### Dino Moves ####

    def move_jump(self):
        self.body.send_keys(Keys.UP)

    def move_duck(self):
        self.body.send_keys(Keys.DOWN)

    def move_pass(self):
        pass # Intentional

    #### RL Functions ####

    def memorize(self,state_info,action):
        self.memory.append(state_info + action)
        return

    def get_action(self):
        raise NotImplementedError

    def replay(self):
        raise NotImplementedError

    #### Run Training Session ####

    def run(self):
        raise NotImplementedError



    



