
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import numpy as np

URL = 'chrome://dino/'
CSS_RUNNER_CONTAINER = '.runner-container'


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
            results.push(runn.currentSpeed);
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

    def __init__(self):
        self.driver = webdriver.Chrome('./chromedriver')
        self.driver.get(self.URL)
        self.body = self.driver.find_element_by_tag_name("body")
        time.sleep(1)
        self.move_jump()
        return

    def __del__(self):
        self.driver.close()

    def get_positions(self):
        return self.driver.execute_script(self.FN)
    
    #### Dino Moves ####

    def move_jump(self):
        self.body.send_keys(Keys.UP)

    def move_duck(self):
        self.body.send_keys(Keys.DOWN)

    def move_pass(self):
        pass # Intentional

    #### Run Training Session ####

    def run(self,n_plays=1):
        THRESHOLD = 100
        time.sleep(3)
        # self.body.send_keys(Keys.UP)
        for i in range(n_plays):
            self.body.send_keys(Keys.UP)
            time.sleep(1)
            done = False
            while not done:
                (
                    done, 
                    dist, 
                    trex_x, 
                    trex_y, 
                    obs1_x, 
                    obs1_y, 
                    obs2_x, 
                    obs2_y,
                    obs3_x, 
                    obs3_y
                ) = self.get_positions()
                if obs1_x - trex_x < THRESHOLD and obs1_y > 80:
                    self.move_jump()
                else:
                    self.move_pass()
                time.sleep(0.05)


    
if __name__ == "__main__":
    print("...PROGRAM STARTING...")
    solver = DinoGame()
    print("...RUNNING...")
    solver.run(25)
    print("...DONE...")
    input("Press anything to close...")


