import numpy as np


class MatMul:
    def __init__(self, W):
        self.params = [W]  # 학습하는 매개변수 저장
        self.grads = [np.zeros_like(W)]  # params에 대응되는 기울기 저장
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matnul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW  # 기울기값 설정(생략 기호 사용) -> 메모리 위치 고정시키고 그 위치에 원소들을 덮어쓴다. (덮어쓰기 지원=깊은 복사)
        return dx
