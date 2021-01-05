'''
1.2.2 계층으로 클래스화 및 순전파 구현
-완전 연결 계층에 의한 변환: Affine 계층
-시그모이드 함수에 의한 변환: Sigmoid 계층

**계층 구현 시**
1) 모든 계층은 forward(), backward() 함수를 가진다. : 순전파, 역전파
2) 모든 계층은 인스턴스 변수인 params(가중치, 편향과 같은 매개변수 담는 리스트)와 grads(해당 매개변수의 기울기 보관 리스트)를 가진다.
*************

'''

import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        return out


'''
구현 신경망 계층 구성
x->Affine->sigmoid->Affine->s
TwoLayerNet 로 추상화하고 주 추론 처리는 predict(x) 함수로 구현
*매개변수들을 하나의 리스트에 보관하면 '매개변수 갱신'과 '매개변수 저장'을 쉽게 할 수 있다.
'''


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 가중치, 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 모든가중치를 리스트에 모음
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)  # 입력
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)

print(s)
