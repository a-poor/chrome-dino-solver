"""
generate_data.py
created by Austin Poor

a file for generating fake data for the model to learn from,
based on the rules of the chrome-dino-game

used by dino_solver_deepQ.py

"""


import numpy as np

J_THRESHOLD = 160
D_THRESHOLD = 75


def get_action(x,y):
    if x < J_THRESHOLD and y > D_THRESHOLD:
        return 1.0
    else:
        return 0.0

def synthesize_rewards(x,y):
    if x < J_THRESHOLD:
        if y > D_THRESHOLD: # Jump over
            a = 1
            r = np.array([-100.0,50.0]) # Jump!
        else: # Fly over
            a = 0
            r = np.array([50.0,-100.0]) # Don't Jump!
    else: # Too far
        a = 0
        r = np.array([0.1,-0.1]) # Doesn't matter
    return r

def generate_point():
    """
    X: [tRexX,tRexY,speed,obsX,obsY]
    Y: [passReward,jumpReward]
    """
    X = np.array([
        np.random.normal(23,1),     # tRex X
        np.random.uniform(0,90),    # tRex Y
        np.random.uniform(6,12),    # speed
        np.random.uniform(-50,700), # obs1 X
        np.random.uniform(0,90)     # obs1 Y
    ])
    y = synthesize_rewards(X[3],X[4])
    return X, y

def generate_data(n_datapoints):
    X, y = zip(*(generate_point() for _ in range(n_datapoints)))
    X = np.array(X)
    y = np.array(y)
    return X, y

