'''
## 8.3 어텐션 평가
날짜 형식을 변경하는 문제: 영어권에서 사용하는 다양한 날짜 형식을 표준 형식으로 변환하는 것이 목표
'''

import sys

from common.util import eval_seq2seq

sys.path.append('..')
sys.path.append('../chap7')
import numpy as np
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from attention_seq2seq import AttentionSeq2seq
from chap7.train_seq2seq import Seq2seq
from chap7.train_seq2seq import PeekySeq2seq
import matplotlib.pyplot as plt


# 데이터 읽기
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 입력 문장 반전
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# 하이퍼파라미터 설정
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('정확도 %.3f%%' % (acc * 100))
