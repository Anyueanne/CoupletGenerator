# -*- coding: utf-8 -*-
# @Time    : 2019/4/13 17:22
# @Author  : MengnanChen
# @FileName: predict.py
# @Software: PyCharm

from six.moves import cPickle as pickle
import tensorflow as tf

batch_size = 50

with open('data/source_map.pik', 'rb') as fin:
    source_int_to_letter, source_letter_to_int = pickle.load(fin)
with open('data/target_map.pik', 'rb') as fin:
    target_int_to_letter, target_letter_to_int = pickle.load(fin)


# 预测
def source_to_seq(text):
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [
        source_letter_to_int['<PAD>']] * (sequence_length - len(text))


input_word = '承上下求索志'
text = source_to_seq(input_word)

checkpoint = 'data/trained_model.ckpt'
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_word)] * batch_size,
                                      source_sequence_length: [len(input_word)] * batch_size})[0]

    pad = source_letter_to_int['<PAD>']

    print('原始输入:', input_word)

    print('\nSource')
    print('  Word 编号:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(' '.join([source_int_to_letter[i] for i in text])))

    print('\nTarget')
    print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format(' '.join([target_int_to_letter[i] for i in answer_logits if i != pad])))
