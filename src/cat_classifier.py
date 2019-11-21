from sklearn import model_selection
import pandas
import pickle
import numpy as np
import time
import yaml
from models import LSTMModel, CATModel

# import spacy
# spacy_en = spacy.load('en')

import torch
import torch.nn as nn
import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets


# dataset = pickle.load(open('data/labelled_dataset.p', 'rb'))
train_dataset = pickle.load(open('../data/labelled_dataset_train.p', 'rb'))
valid_dataset = pickle.load(open('../data/labelled_dataset_valid.p', 'rb'))
test_dataset = pickle.load(open('../data/labelled_dataset_test.p', 'rb'))

# codes_train, asts_train, annos_train, annos2_train = zip(*train_dataset)
# train_dataset = list(zip(codes_train, asts_train, annos_train, annos2_train))
# codes_valid, asts_valid, annos_valid, annos2_valid = zip(*valid_dataset)
# valid_dataset = list(zip(codes_valid, asts_valid, annos_valid, annos2_valid))
# codes_test, asts_test, annos_test, annos2_test = zip(*test_dataset)
# test_dataset = list(zip(codes_test, asts_test, annos_test, annos2_test))

# codes = codes_train + codes_valid + codes_test
# asts = asts_train + asts_valid + asts_test
# annos = annos_train + annos_valid + annos_test + annos2_train + annos2_valid + annos2_test

codes = pickle.load(open('../data/codes','rb'))
annos = pickle.load(open('../data/annos','rb'))
asts = pickle.load(open('../data/asts','rb'))

trainDF = {}
trainDF['code'] = codes
trainDF['anno'] = annos
trainDF['ast'] = asts
# trainDF['label'] = labels




with open("../config.yml", 'r') as config_file:
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)

random_seed = cfg["random_seed"]
np.random.seed(random_seed)
embedding_dim = cfg["embedding_dim"]
learning_rate = cfg["learning_rate"]
seq_len_anno = 0
seq_len_code = 0
hidden_size = cfg["hidden_size"]
dense_dim = cfg["dense_dim"]
output_dim = cfg["output_dim"]
num_layers_lstm = cfg["num_layers_lstm"]
use_cuda = cfg["use_cuda"]
batch_size = cfg["batch_size"]
# n_iters = 4000
# num_epochs = n_iters / (len(train_dataset) / batch_size)
# num_epochs = int(num_epochs)
num_epochs = cfg["epochs"]
use_softmax_classifier = cfg["use_softmax_classifier"]
use_bin = cfg["use_bin"]
use_bidirectional = cfg["use_bidirectional"]
use_adam = cfg["use_adam"]
use_parallel = cfg["use_parallel"]
save_path = cfg["save_path"]
if use_cuda:
    device_id = 0
    torch.cuda.set_device(device_id)

print("Number of epochs = ", num_epochs)



# Loading word embeddings
if use_bin:
    import fastText.FastText as ft
    ft_anno_vec = ft.load_model('conala/ft_models/anno_model.bin')
    ft_code_vec = ft.load_model('conala/ft_models/code_model.bin')
else:
    from keras.preprocessing import text, sequence

# def tokenizer(text): # create a tokenizer function
#     return [tok.text for tok in spacy_en.tokenizer(text)]


def prepare_sequence(seq, seq_len, to_ix):
    idxs_list = []
    
    for seq_elem in seq:
        idxs = []
        for w in seq_elem.split():
            try:
                idxs.append(to_ix[w])
            except KeyError:
                continue
        # idxs = [to_ix[w] for w in seq_elem.split()]
        # idxs = [to_ix[w] for w in tokenizer(seq_elem)]
        idxs.reverse()
        if len(idxs) > seq_len:
            idxs = idxs[:seq_len]
        while len(idxs) < seq_len:
            idxs.append(0)
        idxs.reverse()
        idxs_list.append(idxs)
    return torch.tensor(idxs_list, dtype=torch.long)


def create_embeddings(fname, embed_type):
    embeddings_index = {}
    for i, line in enumerate(open(fname)):
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

    # create a tokenizer
    token = text.Tokenizer(char_level=False)
    token.fit_on_texts(trainDF[embed_type])
    word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors
    # train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=seq_len)
    # valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=seq_len)

    # create token-embedding mapping
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return word_index, embedding_matrix


# Create word-index mapping


word_to_ix_anno = {}
word_to_ix_code = {}

if use_bin:
    count = 0
    count1 = 0
    for code, anno, lab in dataset:
        anno_list = anno.split()
        sent_length_anno = len(anno_list)
        count += 1
        if sent_length_anno > seq_len_anno:
            seq_len_anno = sent_length_anno
        for word in anno_list:
            if word not in word_to_ix_anno:
                word_to_ix_anno[word] = len(word_to_ix_anno) + 1
        code_list = code.split()
        sent_length_code = len(code_list)
        if sent_length_code > seq_len_code:
            seq_len_code = sent_length_code
        if sent_length_code > 40:
            count1 += 1
        for word in code_list:
            if word not in word_to_ix_code:
                word_to_ix_code[word] = len(word_to_ix_code) + 1

    seq_len_code = 40

    matrix_len = len(word_to_ix_anno) + 1  # +1 because of padding
    weights_matrix_anno = np.zeros((matrix_len, embedding_dim))

    for word in word_to_ix_anno:
        weights_matrix_anno[word_to_ix_anno[word]] = ft_anno_vec.get_word_vector(word)

    matrix_len = len(word_to_ix_code) + 1  # +1 because of padding
    weights_matrix_code = np.zeros((matrix_len, embedding_dim))

    for word in word_to_ix_code:
        weights_matrix_code[word_to_ix_code[word]] = ft_code_vec.get_word_vector(word)

else:
    seq_len_code = seq_len_anno = seq_len_ast = 300
    word_to_ix_anno, weights_matrix_anno = create_embeddings('../saved_models/anno_model.vec', 'anno')
    word_to_ix_code, weights_matrix_code = create_embeddings('../saved_models/code_model.vec', 'code')
    word_to_ix_ast, weights_matrix_ast = create_embeddings('../saved_models/ast_model.vec', 'ast')


weights_matrix_anno = torch.from_numpy(weights_matrix_anno)
weights_matrix_code = torch.from_numpy(weights_matrix_code)
weights_matrix_ast = torch.from_numpy(weights_matrix_ast)


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

sim_model = CATModel(weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code,
    weights_matrix_ast)


if torch.cuda.is_available() and use_cuda:
    # anno_model.cuda()
    # code_model.cuda()
    sim_model.cuda()
    if use_parallel:
        anno_model = nn.DataParallel(anno_model)
        code_model = nn.DataParallel(code_model)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=1,
#                                           shuffle=False)


if use_softmax_classifier:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()

# optimizer = torch.optim.Adam(list(anno_model.parameters()) + list(code_model.parameters()), lr=learning_rate)
if use_adam:
    opt = torch.optim.Adam(sim_model.parameters(), lr=learning_rate)
    # opt1 = torch.optim.Adam(anno_model.parameters(), lr=learning_rate)
    # opt2 = torch.optim.Adam(code_model.parameters(), lr=learning_rate)

else:
    opt1 = torch.optim.SGD(anno_model.parameters(), lr=learning_rate, momentum=0.9)
    opt2 = torch.optim.SGD(code_model.parameters(), lr=learning_rate, momentum=0.9)
    


# Training
iter = 0
sim_model.train()
start_time = time.time()
for epoch in range(num_epochs):
    epoch += 1
    batch_iter = 0
    for i, (code_sequence, ast_sequence, anno_sequence, anno_sequence_neg) in enumerate(train_loader):
        sim_model.zero_grad()
        anno_in = prepare_sequence(anno_sequence, seq_len_anno, word_to_ix_anno)
        code_in = prepare_sequence(code_sequence, seq_len_code, word_to_ix_code)
        ast_in = prepare_sequence(ast_sequence, seq_len_code, word_to_ix_ast)
        anno_in_neg = prepare_sequence(anno_sequence_neg, seq_len_anno, word_to_ix_anno)
        if torch.cuda.is_available() and use_cuda:
            sim_score, _, _ = sim_model(anno_in.cuda(), code_in.cuda(), ast_in.cuda())
            sim_score_neg, _, _ = sim_model(anno_in_neg.cuda(), code_in.cuda(), ast_in.cuda())
        else:
            sim_score, _, _ = sim_model(anno_in, code_in, ast_in)
            sim_score_neg, _, _ = sim_model(anno_in_neg, code_in, ast_in)
        
        loss =  0.05 - sim_score + sim_score_neg
        loss[loss<0] = 0.0
        loss = torch.sum(loss)
        loss.backward()
        opt.step()
        iter += 1
        batch_iter += 1
        print("Epoch: {}. Iteration: {}. Loss: {}".format(epoch, batch_iter, loss))

print('Time taken to train: {} seconds'.format(time.time()-start_time))
print("Saving Models")
torch.save(sim_model.state_dict(), f"../{save_path}/sim_model_ast")
print("Saved Models")