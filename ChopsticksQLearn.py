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
        if np.sum(theta[i, :]) == 0:  # theta의 합이 0인 경우
            pi[i, :] = np.ones(n) / n  # 모든 원소에 동일한 확률 부여
        else:
            pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

def get_state_number(state):
    return state[0, 0] * 1 + state[0, 1] * 6 + state[1, 0] * 36 + state[1, 1] * 216

def get_next_state(pi, state):
    action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    act_number = get_state_number(state)
    if np.random.rand()
    get_action = np.random.choice(action, p=pi[act_number, :])
    if get_action == 0:
        np.put(state, [1,0], state[0,0] + [1,0])
    elif get_action == 1:
        np.put(state, [1,1], state[0,0] + [1,1])
    elif get_action == 2:
        np.put(state, [1,0], state[0,1] + [1,0])
    elif get_action == 3:
        np.put(state, [1,1], state[0,1] + [1,1])
    elif get_action >= 4 and get_action <= 8:
        np.put(state, [0,0], state[0,0] + (get_action - 3))
        np.put(state, [0,1], state[0,1] - (get_action - 3))
    elif get_action >= 9 and get_action <= 13:
        np.put(state, [0,0], state[0,0] - (get_action - 8))
        np.put(state, [0,1], state[0,1] + (get_action - 8))
    if np.any(state > 5):
        state[state > 5] = 0
    return state

def reverse_state(state):
    return np.array([state[1, :],state[0, :]])

def env(pi):
    state = np.array([[0, 0], [0, 0]])
    state_history = np.array([[[0,0],[0,0]]])
    turn = 0
    while (1):
        turn += 1
        if turn % 2 == 1:
            next_state = get_next_state(pi, state)
            state_history = np.concatenate((state_history, [next_state]), axis=0)
            s = next_state
        else:
            next_state = reverse_state(get_next_state(pi, reverse_state(state)))
            state_history = np.concatenate((state_history, [next_state]), axis=0)
            s = next_state
        
        if (state[0, :] != [0, 0]).all() and (state[1, :] != [0, 0]).all():
            break
    return state_history


theta = getFullTheta()
pi_0 = convert_theta_into_pi(theta)
print(theta)
print(theta.shape)



state_history = env(pi_0)
print(state_history)
np.save('C:/Users/sunwo/Documents/GitHub/ChopsitcksMaster/theta', theta)