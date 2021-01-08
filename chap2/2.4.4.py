'''
2.4.4 PTB 데이터셋
'''

import sys

sys.path.append('..')
import ptb
import numpy as np

corpus, word_to_id, id_to_word = ptb.load_data('train')  # train, test, valid 중 하나 선택 가능

print('말뭉치 크기:', len(corpus))
print('corpus[:30]:', corpus[:30])
print()
print('id_to_word[0]:', id_to_word[0])
print('id_to_word[1]:', id_to_word[1])
print('id_to_word[2]:', id_to_word[2])
print()
print("word_to_id['car']:", word_to_id['car'])
print("word_to_id['happy']:", word_to_id['happy'])
print("word_to_id['lexus']:", word_to_id['lexus'])
