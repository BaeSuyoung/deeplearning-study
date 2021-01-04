'''
1.2.1 신경망 추론
'''

import numpy as np


# 활성화 함수로 sigmoid 함수 사용: 0~1사이 실수 출력
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.random.randn(10, 2)  # 입력층(2차원 데이터 10개 미니배치로 처리)
w1 = np.random.randn(2, 4)  # 입력층(2)~은닉층(4) 사이 가중치
b1 = np.random.randn(4)  # bias
w2 = np.random.randn(4, 3)  # 은닉층(4)~출력층(3) 사이 가중치
b2 = np.random.randn(3)  # bias

h = np.matmul(x, w1) + b1
a = sigmoid(h)
s = np.matmul(a, w2) + b2

print(h)
print(s)  # 최종 결과 각 데이터가 3차원 데이터로 변환된다. =점수

'''
점수: 확률이 되기 전 값. 점수를 소프트맥스 함수에 입력하면 함수를 얻을 수 있다.
'''
