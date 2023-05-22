import numpy as np
from nltk.tokenize import word_tokenize
from rnn import RNN
import string

def get_vocabulary_and_mappings (chars):
    vocabulary = list(set(chars))
    vocabulary.sort()
    print("Vocabulary size: ", len(vocabulary))
    
    index_to_char = {}
    char_to_index = {}
    
    for idx, char in enumerate(vocabulary):
        index_to_char[idx] = char
        char_to_index[char] = idx
    
    return vocabulary, index_to_char, char_to_index

def get_one_hot(index, size):
    one_hot = np.zeros([size, 1])
    one_hot[index] = 1
    return one_hot

def indexes_to_sample(samples, index_to_word):
    words = [index_to_word[index.argmax().item()] for index in samples]
    return ' '.join(words)

def get_batch(chars, char_to_index, timestamps, V):
    for index in range(0, len(chars) - timestamps - 1, timestamps):
        chars_in = chars[index : index + timestamps]
        chars_out = chars[index + 1 : index + timestamps + 1]
        
        x = [get_one_hot(char_to_index[char], V) for char in chars_in]
        y = [get_one_hot(char_to_index[char], V) for char in chars_out]
        
        yield chars_in, np.array(x).T[0], np.array(y).T[0]

def sample_sequence(input_char, h, char_to_index):
    last_char = input_char
    samples = [last_char]
    
    for i in range(350):
        one_hot = get_one_hot(char_to_index[last_char], V)
        y_preds = rnn.predict(one_hot, h)
        probabilities = y_preds[0]
        ix = np.random.choice(range(V), size=1, p=probabilities.ravel())
        last_char = index_to_char[ix.item()]
        samples.append(last_char)
        
        h = rnn.model[0].h

    return ''.join(samples)

corpus = open('./tiny-shakespeare.txt', 'r').read()
chars = [char for char in corpus if char.isalpha() or char in string.punctuation or char =='\n' or char == ' ']
vocabulary, index_to_char, char_to_index = get_vocabulary_and_mappings(chars)
V = len(vocabulary)
timestamps = 20

rnn = RNN(
    n_in = V,
    n_hidden = 256,
    n_out = V,
    n_timestamps = timestamps
)

epochs = 50
alpha = 0.1

for epoch in range(epochs):
    total_cost = 0
    n_iter = 1
    for input_sequence, x, y in get_batch(chars, char_to_index, timestamps, V):
        cost = rnn.train(x, y, alpha)
        total_cost += cost
        
        if n_iter % 10000 == 0:
            print(f"iter: {n_iter} cost: {total_cost/ n_iter:.6f}")
        
        if n_iter % 20000 == 0:
            ix = np.random.randint(0, timestamps).item()
            print("=============")
            print("Sample input", input_sequence[ix])
            print("Sampled output:", sample_sequence(input_sequence[ix], rnn.model[ix].h, char_to_index))
            print("=============")
            
        n_iter += 1

    if epoch % 15 == 0:
        alpha *= 0.66
        np.savez_compressed('/home/shakespeare-char-weights', Wxh=rnn.Wxh, Whh=rnn.Whh, bh=rnn.bh, Why=rnn.Why, by=rnn.by)

    print(f"Epoch: {epoch+1} cost: {total_cost/ n_iter:.6f}")

