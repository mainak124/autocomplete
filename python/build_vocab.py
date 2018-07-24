import argparse

parser = argparse.ArgumentParser(description='Build character vocabulary from input text file.')
parser.add_argument('-i', '--input', default='data/history.txt', required=True, help='Input text file to create vocabulary')
parser.add_argument('-v', '--vocab', default='data/vocab.txt', required=True, help='Output vocabulary file')

args = parser.parse_args()

data_file_name = args.input
vocab_file_name = args.vocab

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
