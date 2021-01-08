'''
2.4.1 PPMI 구현
'''

import numpy as np
import matplotlib.pyplot as plt


########################################
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):  # 단어 id 리스트, 어휘 수, 윈도우 크기
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)
    return np.dot(nx, ny)


################################################################
'''
C: 동시 발생 행렬
berbose : 진행상황 출력 여부

'''


def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if (verbose):
                cnt += 1
                if cnt % (total // 100) == 0:
                    print('%.1f%% 완료' % (100 * cnt / total))
    return M


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

np.set_printoptions(precision=3)  # 유효 자릿수를 세 자리로 표시
print("동시발생 행렬")
print(C)
print('-' * 50)
print("PPMI")
print(W)
print('-' * 50)
print("SVD")
print(U)

## 2차원 벡터로 줄이려면 처음 두 원소 꺼내면 된다.
print('-' * 50)
print(U[0, :2])

##그래프로 그려보기
for word, word_id in word_to_id.items():
    plt.annotate(word, U[word_id, 0], U[word_id, 1])
plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
