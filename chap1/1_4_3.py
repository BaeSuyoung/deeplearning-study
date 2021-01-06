'''
1.4.3 학습용 코드
-학습 데이터 읽어들여 신경망과 옵티마이저 생성
1. hyperparameter 설정(변수)
2. 데이터 읽기, 모델, 옵티마이저 설정
3. 데이터 섞기
4. 기울기 구하고 매개변수 갱신
5. 정기적으로 학습 결과 출력
-------------------------------------------------------------------------------


max_epoch = 300  # 학습 단위 (데이터 300번 반복한다는 소리.)
batch_size = 30  # 1 에톡 당 30개 데이터를 랜덤 선택해서 학습한다.
hidden_size = 10
learning_rate = 1.0

# 2. 데이터 읽기
x, t = load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# 학습에 사용되는 변수
data_size = len(x)  # 데이터 길이
max_iters = data_size // batch_size  # 300/30=10 최대 10번 반복
total_loss = 0
loss_count = 0
loss_list = []
epoch_num = []

for epoch in range(max_epoch):  # 300번
    # 3. 데이터 섞기
    idx = np.random.permutation(data_size)  # data_size-1 만큼 무작위로 섞는다.
    x = x[idx]
    t = t[idx]
    epoch_num.append(epoch + 1)

    for iters in range(max_iters):  # 10번
        batch_x = x[iters * batch_size:(iters + 1) * batch_size]  # 앞에서부터 순서대로 뽑는다.
        batch_t = t[iters * batch_size:(iters + 1) * batch_size]

        # 4. 기울기 구해 매개변수 갱신
        loss = model.forward(batch_x, batch_t)  # 손실 값 구하기
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 5. 정기적으로 학습 경과 출력
        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| 에폭 %d | 반복 %d / %d | 손실 %.2f' % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

plt.plot(epoch_num, loss_list)
plt.show()
'''

'''1.4.4 Triner Class '''

import sys

sys.path.append('')
import time
import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet


def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # 클래스당 샘플 수
    DIM = 2  # 데어터 요소 수
    CLS_NUM = 3  # 클래스 수

    x = np.zeros((N * CLS_NUM, DIM))  # 테스트 값
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=np.int)  # 결과 값

    for j in range(CLS_NUM):
        for i in range(N):  # N*j, N*(j+1)):
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t


class SGD:
    '''
    확률적 경사하강법(Stochastic Gradient Descent)
    '''

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Trainer:
    # 1. hyperparameter
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=30, max_grad=None, eval_interval=20):
        data_size = len(x)  # 데이터 길이
        max_iters = data_size // batch_size  # 300/30=10 최대 10번 반복
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()

        for epoch in range(max_epoch):  # 300번
            # 3. 데이터 섞기
            idx = np.random.permutation(data_size)  # data_size-1 만큼 무작위로 섞는다.
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):  # 10번
                batch_x = x[iters * batch_size:(iters + 1) * batch_size]  # 앞에서부터 순서대로 뽑는다.
                batch_t = t[iters * batch_size:(iters + 1) * batch_size]

                # 4. 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)  # 손실 값 구하기
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(model.params, model.grads)

                total_loss += loss
                loss_count += 1

                # 5. 정기적으로 학습 경과 출력
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('손실')
        plt.show()


def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                        params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


###### 메인 함수 ######

max_epoch = 300  # 학습 단위 (데이터 300번 반복한다는 소리.)
batch_size = 30  # 1 에톡 당 30개 데이터를 랜덤 선택해서 학습한다.
hidden_size = 10
learning_rate = 1.0

# 2. 데이터 읽기
x, t = load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()
