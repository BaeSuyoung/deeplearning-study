'''
1.3.5 기울기 도출과 역전파 구현
- sigmoid, Affine, Softmax with Loss 계층 구현
'''

import numpy as np


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:  # MsyMul 계층 사용하면 편하다.
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.params[1][...] = db
        return dx


'''
softmax with Loss 계층
-softmax의 출력 (y1, y2, y3) 와 정답 레이블(t1, t2, t3) 를 받고 손실 L을 구한다. (y-t)

'''

'''
1.3.6 가중치 갱신
- 기울기 사용해서 매개변수 갱신
1. 미니배치: 훈련 데이터 중 무작위로 다수의 데이터 골라낸다.
2. 기울기 계산: 오차 역전파법으로 각 가중치 매개변수에 대한 손실함수의 기울기를 구한다.
3. 매개변수 갱신 -> 경사 하강법(Gradient Descent)
4. 반복 (1~3)

**가중치 갱신 기법: 확률적 경사 하강법(Stochastic Gradient Descent: SGD):  무작위로 선택된 데이터에 대한 기울기 사용
: 가중치를 기울기 방향으로 일정 거리만큼 갱신한다.
: W= W- (Learning Rate)*(기울기)
: Learning Rate: 0.01~0.001 미리 정해둠.
'''


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

