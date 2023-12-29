import numpy as np
import time
import random

'''
함수 구현
'''

# 초기 정책 구현

# state 받으면 정책 하나 만들기
def getSingleTheta(state):
    theta = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
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
    for i in range(1, 5): # 빼기가 가능한 경우
        if a - i >= 0 and b + i >= 4 and [a - i, b + i] != [b, a]:
            np.put(theta, [i+3], 1)
    for i in range(1, 5): # 더하기가 가능한 경우
        if a + i <= 4 and b - i >= 0 and [a + i, b - i] != [b, a]:
            np.put(theta, [i+7], 1)
    return theta

# 모든 정책 나열
def getFullTheta():
    state = np.array([[1, 0], [0, 0]])
    theta = np.array([[0,0,0,0,0,0,0,0,0,0,0,0]])
    while True:
        single_theta = getSingleTheta(state)
        theta = np.vstack((theta, single_theta))
        if state[0][0] < 4:
            np.put(state, [0][0], state[0][0]+1)
        elif state[0][1] < 4:
            np.put(state, [0], 0)
            np.put(state, [1], state[0][1]+1)
        elif state[1][0] < 4:
            np.put(state, [0], 0)
            np.put(state, [1], 0)
            np.put(state, [2], state[1][0]+1)
        elif state[1][1] < 4:
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

    state_number = get_state_number(state)
    state_next_number = get_state_number(state_next)

    # 정책 의존성 없는 Q 러닝
    Q[state_number, action] = Q[state_number, action] + eta * (reward + gamma * np.nanmax(Q[state_next_number, :]) - Q[state_number, action])

    return Q

# epsilon-greedy 알고리즘

def get_state_number(state): # 몇번째 정책인지 구하기
    return state[0, 0] * 1 + state[0, 1] * 5 + state[1, 0] * 25 + state[1, 1] * 125

def get_action(state, Q, epsilon, pi_0): # action 구하기
    action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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
    s_next = state.copy()
    if action == 0:
        s_next[1, 0] += s_next[0, 0]
    elif action == 1:
        s_next[1, 1] += s_next[0, 0]
    elif action == 2:
        s_next[1, 0] += s_next[0, 1]
    elif action == 3:
        s_next[1, 1] += s_next[0, 1]
    elif action >= 4 and action <= 7:
        s_next[0, 0] -= action - 3
        s_next[0, 1] += action - 3
    elif action >= 8 and action <= 11:
        s_next[0, 0] += action - 7
        s_next[0, 1] -= action - 7
    if np.any(s_next > 4):
        s_next[s_next > 4] = 0
    return s_next

def env(Q, epsilon, eta, gamma, pi): # 1대1 환경
    state = np.array([[1, 1], [1, 1]]) # 초기 상태
    turn = 1 # 턴 수

    while(1): # 무한 루프
        is_player1 = turn % 2 == 1 # 1P or 2P 판별

        # 플레이어 입장에서 상태 변환
        state_perspective = state if is_player1 else state[::-1]

        # 행동 결정
        action = get_action(state_perspective, Q, epsilon, pi) 
        
        # 다음 단계 state 구하기
        state_next_perspective = get_next_state(state_perspective, action)

        # 다음 상태를 실제 상태에 반영
        state_next = state_next_perspective if is_player1 else state_next_perspective[::-1]

        # 보상 부여 및 다음 행동 계산
        if (state_next_perspective[1, 0] == 0 and state_next_perspective[1, 1] == 0): # 이긴 경우
            reward = 1
        elif (state_next_perspective[0, 0] == 0 and state_next_perspective[0, 1] == 0): # 진 경우
            reward = -1
        else:
            reward = 0

        
        if reward == 1: # 이긴 경우 이전 행동 가치 함수 보상 -1
            Q = Q_learning(state_perspective_last, action_last, -1, state_next_perspective_last, Q, eta, gamma)

        # 가치함수 수정
        Q = Q_learning(state_perspective, action, reward, state_next_perspective, Q, eta, gamma)
        
        # 모두 0 또는 그 이하일 경우
        
        if np.nanargmax(Q[get_state_number(state_perspective), :]) <= 0:
            Q[get_state_number(state_perspective), :] = pi[get_state_number(state_perspective), :]

        # 종료 여부 판정
        if (state_next[0, 0] == 0 and state_next[0, 1] == 0) or (state_next[1, 0] == 0 and state_next[1, 1] == 0):
            break
        else:
            state_perspective_last = state_perspective.copy()
            state_next_perspective_last = state_next_perspective.copy()
            action_last = action
            state = state_next
        turn += 1 # 턴 수 증가

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
Q = np.random.rand(a, b) * theta_0

# 초기 설정
eta = 0.01 # 학습률
gamma = 0.01 # 시간할인률
epsilon = 0.9 # 무작위 값을 취할 확률
v = np.nanmax(Q, axis=1) # 각 상태마다 가치의 최댓값 계산
is_continue = True # 루프용
episode = 1 # 에피소드 수
V = [] # 각 에피소드별로 상태가지 저장
V.append(np.nanmax(Q, axis=1)) # 상태별로 행동가치의 최댓값 계산

while is_continue:
    print("에피소드 " + str(episode))

    # epsilon 값 감소
    epsilon = max(0.01, epsilon - 0.01)

    # 턴 수와 Q함수 저장
    [turn, Q] = env(Q, epsilon, eta, gamma, pi_0)

    # 상태가치 변화값 계산
    new_v = np.nanmax(Q, axis=1)
    change = np.sum(np.abs(new_v - v))

    # 출력
    print("상태가치 변화: " + str(change))
    print("걸린 턴 수: " + str(turn))

    v = new_v

    # 반복
    episode += 1
    if change < 0.002 and episode > 10000:
        break

np.save('C:/Users/user/Documents/GitHub/ChopsitcksMaster/Q', Q)
print('성공적으로 가치함수를 저장하였습니다')