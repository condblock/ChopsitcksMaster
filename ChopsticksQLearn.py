import numpy as np


'''
함수 구현
'''

# 초기 정책 구현

# state 받으면 정책 하나 만들기
def getSingleTheta(state):
    theta = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
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
        if a + i <= 5 and b - i >= 0:
            np.put(theta, [i+4], 1)
    for i in range(0, min(5-b, a)): # 더하기가 가능한 경우
        if a - i >= 0 and b + i <= 5:
            np.put(theta, [i+9], 1)
    return theta

# 모든 정책 나열
def getFullTheta():
    state = np.array([[0, 0], [0, 0]])
    theta = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    while True:
        single_theta = getSingleTheta(state)
        theta = np.vstack((theta, single_theta))
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

def convert_theta_into_pi(theta): # 정책 모두 0, 1로 되어있는것 비율로 수정
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        if np.sum(theta[i, :]) == 0:  # theta의 합이 0인 경우
            pi[i, :] = np.ones(n) / n  # 모든 원소에 동일한 확률 부여
        else:
            pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

# Q 러닝 함수

def Q_learning(state, action, reward, state_next, Q, eta, gamma):

    # 정책 의존성 없는 Q 러닝
    if (state[1, :] == [0, 0]).all(): # 이겼을 때
        Q[state, action] = Q[state, action] + eta * (reward - Q[state, action])
    else:
        Q[state, action] = Q[state, action] + eta * (reward + gamma * np.nanmax(Q[state_next, :]) - Q[state, action])

    return Q

# epsilon-greedy 알고리즘

def get_state_number(state): # 몇번째 정책인지 구하기
    return state[0, 0] * 1 + state[0, 1] * 6 + state[1, 0] * 36 + state[1, 1] * 216

def get_action(state, Q, epsilon, pi_0): # action 구하기
    action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    act_number = get_state_number(state)
    # 행동 결정
    if np.random.rand() < epsilon:
        # 확률 epsilon으로 무작위 행동 선택
        next_action = np.random.choice(action, p=pi_0[act_number, :])
    else:
        # Q값이 최대가 되는 행동 선택
        next_action = action[np.nanargmax(Q[act_number, :])]
    return next_action

def get_next_state(state, action): # 다음 state 구하기
    if action == 0:
        state[1, 0] += state[0, 0]
    elif action == 1:
        state[1, 1] += state[0, 0]
    elif action == 2:
        state[1, 0] += state[0, 1]
    elif action == 3:
        state[1, 1] += state[0, 1]
    elif action >= 4 and action <= 8:
        state[0, 0] += action - 3
        state[0, 1] -= action - 3
    elif action >= 9 and action <= 13:
        state[0, 0] -= action - 8
        state[0, 1] += action - 8
    if np.any(state > 5):
        state[state > 5] = 0
    return state

def reverse_state(state): # state 전환
    return np.array([state[1, :],state[0, :]])

def env(Q, epsilon, eta, gamma, pi): # 환경
    state = np.array([[1, 1], [1, 1]]) # 초기 상태
    turn = 0 # 턴 수
    action = action_next = get_action(state, Q, epsilon, pi) # 초기 행동

    while (1):
        action = action_next # 행동 결정
        turn += 1 # 턴 수

        # 다음 단계 state 구하기
        state_next = get_next_state(state, action) 

        # 보상 부여 후 다음 행동 계산
        if (state[1, :] == [0, 0]).all(): # 이긴 경우
            reward = 1
            action_next = np.nan
        else:
            # 플레이어의 시점에 따라 state 뒤집기
            if turn % 2 == 0: 
                state = reverse_state(state)
            reward = 0
            action_next = get_action(state_next, Q, epsilon, pi)

        # 가치함수 수정
        Q = Q_learning(state, action, reward, state_next, Q, eta, gamma)


        # 종료 여부 판정
        if (state[0, :] == [0, 0]).all() or (state[1, :] == [0, 0]).all():
            break
        else:
            state = state_next
    return [turn, Q]


'''
학습 코드
'''

# 초기 정책 계산
theta_0 = getFullTheta()
theta = theta_0
pi_0 = convert_theta_into_pi(theta)

# 초기 Q 정의
[a, b] = theta_0.shape
Q = np.random.rand(a, b) * theta_0 * 0.1

# 초기 설정
eta = 0.1
gamma = 0.9
epsilon = 0.5
v = np.nanmax(Q, axis=1)
is_continue = True
episode = 1
V = []
V.append(np.nanmax(Q, axis=1))

while is_continue:
    print("에피소드 " + str(episode))

    epsilon /= 2

    [turn, Q] = env(Q, epsilon, eta, gamma, pi_0)

    new_v = np.nanmax(Q, axis=1)
    print("상태가치 변화" + str(np.sum(np.abs(new_v - v))))
    v = new_v
    print("걸린 턴 수: " + str(turn))

    episode += 1
    if episode > 100:
        break
np.save('C:/Users/sunwo/Documents/GitHub/ChopsitcksMaster/Q', Q)
