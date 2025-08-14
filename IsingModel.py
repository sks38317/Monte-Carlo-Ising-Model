import numpy as np

import numba as nb

import matplotlib.pyplot as plt

from matplotlib import colors

cmap = colors.ListedColormap(['yellow', 'blueviolet'])



k = 180            # k * k개의 상태 격자

epochs = 101      # 시뮬레이션 횟수

J = 1.9

kT = 4

ns = k*k           # ns개의 spin을 업데이트

State = np.random.choice([1,-1], [k, k])  # 초기 spin 값 (+1 or -1)



# 몬테카를로 이징 모형

# ----------------------------------

@nb.jit(nopython=True)

def changeState(state):

    for i in range(ns):

            # 임의 지점의 state를 선택해서 조건에 따라 state를 flip함.

                    n = np.random.randint(0, k)
                    m = np.random.randint(0, k)
                    S = state[n, m]

                    # S와 인접한 neighborhood state들의 합계를 계산함.

                    nbState = state[(n + 1) % k, m] + \
                    state[n, (m+1) % k] + \
                    state[(n-1) % k, m] + \
                    state[n,(m-1) % k]

                    #=====================================================================

                    # S의 state를 flip해 보고 dE에 따라 flip할 것인지 결정
                    # dH < 0이면 flip을 허용하고, dH > 0이면 exp(-dH / kT)의 확률로 flip을 허용

                    H1 = -J * nbState * S
                    H2 = -J * nbState * (S * (-1))

                    dH = H2 - H1

                    if dH < 0:
                      state[n, m] *= -1

                    else:
                      prob = np.exp(-dH / kT)
                    if np.random.random() <= prob:
                      state[n, m] *= -1
    return state

for t in range(epochs):
  State = changeState(State)

  # k * k로 이루어진 격자의 상태를 시각화

  if t % 2 == 0:                  #실행한 시뮬레이션 표시 간격
    pos = str((State > 0).sum())
    neg = str((State < 0).sum())
    plt.figure(figsize=(6,6))
    plt.imshow(State, cmap=cmap)
    plt.title("iteration = " + str(t) + ", (+1) = " + pos + ", (-1) = " + neg)
    plt.ion()
