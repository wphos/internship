from hash_center import generate_hash_center

num_of_classes = 200
code_lengths = [48]

for code_length in code_lengths:
    centers = generate_hash_center(num_of_classes, code_length)
