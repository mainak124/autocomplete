data_file_name = 'history.txt'
vocab_file_name = 'vocab.txt'

f_d = open(data_file_name, 'r')
f_v = open(vocab_file_name, 'w')

f_v.write('<pad>\n<s>\n</s>\n')

vocab = set()

for line in f_d:
    vocab.update(line.strip())

for c in vocab:
    f_v.write(c+'\n')

f_d.close()
f_v.close()
