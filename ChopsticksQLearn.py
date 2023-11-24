import numpy as np

def getSingleTheta(state):
    theta = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    a = state[0, 0]
    b = state[0, 1]
    c = state[1, 0]
    d = state[1, 1]
    if [a, b] == [0, 0] or [c, d] == [0, 0]:
        return theta
    # 공격
    if a != 0:
        if c != 0:
            np.put(theta, [0], 1)
        if d != 0:
            np.put(theta, [1], 1)
    if b != 0:
        if c != 0:
            np.put(theta, [2], 1)
        if d != 0:
            np.put(theta, [3], 1)
    # 교환
    for i in range(0, min(5-a, b)): # 빼기가 가능한 경우
        np.put(theta, [i+4], 1)
    for i in range(0, min(5-b, a)): # 더하기가 가능한 경우
        np.put(theta, [i+9], 1)
    return theta

def getFullTheta():
    state = np.array([[1, 0], [0, 0]])
    theta = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    while True:
        single_theta = np.array([getSingleTheta(state)])
        theta = np.concatenate((theta, single_theta),axis=0)
        if state[0][0] < 5:
            np.put(state, [0][0], state[0][0]+1)
        elif state[0][1] < 5:
            np.put(state, [0], 0)
            np.put(state, [1], state[0][1]+1)
        elif state[1][0] < 5:
            np.put(state, [0], 0)
            np.put(state, [1], 0)
            np.put(state, [2], state[1][0]+1)
        elif state[1][1] <  5:
            np.put(state, [0], 0)
            np.put(state, [1], 0)
            np.put(state, [2], 0)
            np.put(state, [3], state[1][1]+1)
        else:
            break
    return theta

def convert_theta_into_pi(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        if (theta[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            pi[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

def get_next_s(pi, state):
    action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    get_action = np.random.choice(action, p=pi[state, :])
    if get_action == 0:
        np.put(state, [1,0], state[0,0] + [1,0])
    elif get_action == 1:
        np.put(state, [1,1], state[0,0] + [1,1])
    elif get_action == 2:
        np.put(state, [1,0], state[0,1] + [1,0])
    elif get_action == 3:
        np.put(state, [1,1], state[0,1] + [1,1])
    elif get_action >= 4 and get_action <= 8:
        np.put(state, [0,0], state[0,0] - (get_action - 3))
        np.put(state, [0,1], state[0,1] + (get_action - 3))
    elif get_action >= 9 and get_action <= 13:
        np.put(state, [0,0], state[0,0] + (get_action - 8))
        np.put(state, [0,1], state[0,1] - (get_action - 8))
    if np.any(state > 5):
        state[state > 5] = 0
    return 

def env(pi):
    state = np.array([[0, 0], [0, 0]])
    state_history = [0]
    while state[1, :] == [0, 0]:

    
theta = getFullTheta()
theta = convert_theta_into_pi(theta)
print(theta)
print(theta.shape)

np.save('C:/Users/user/Documents/GitHub/ChopsitcksMaster/theta', theta)