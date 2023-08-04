import numpy as np
from hash_center import min_hamming_distance

num_classes = 200
code_lengths = [48]

for code_length in code_lengths:
    h = np.loadtxt('centers/center_%s_%s.txt' % (num_classes, code_length))
    print(np.all((h == 1) | (h == -1)))
    print('num_classes:%s, code_length:%s, min_dis:%s' %
          (num_classes, code_length, min_hamming_distance(h)))
