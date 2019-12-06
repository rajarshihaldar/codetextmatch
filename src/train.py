from sklearn import model_selection
import pandas
import pickle
import numpy as np
import time
import json
import yaml
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.utils.data
from keras.preprocessing import text, sequence
from models import MPCTMClassifier, CATModel, CTModel

with open("../config.yml", 'r') as config_file:
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)

if cfg["dataset"] == "codesearchnet":
    train_dataset = pickle.load(open('../data/labelled_dataset_train.p', 'rb'))
    valid_dataset = pickle.load(open('../data/labelled_dataset_valid.p', 'rb'))
    test_dataset = pickle.load(open('../data/labelled_dataset_test.p', 'rb'))

    if cfg["model"] == 'ct':
        codes_train, _, annos_train, annos2_train = zip(*train_dataset)
        train_dataset = list(zip(codes_train, annos_train, annos2_train))
        codes_valid, _, annos_valid, annos2_valid = zip(*valid_dataset)
        valid_dataset = list(zip(codes_valid, annos_valid, annos2_valid))
        codes_test, _, annos_test, annos2_test = zip(*test_dataset)
        test_dataset = list(zip(codes_test, annos_test, annos2_test))

    codes = pickle.load(open('../data/codes','rb'))
    annos = pickle.load(open('../data/annos','rb'))
    asts = pickle.load(open('../data/asts','rb'))


elif cfg["dataset"] == 'conala':
    train_dataset = pickle.load(open('../../data_conala/conala_labelled_dataset_train.pkl', 'rb'))
    test_dataset = pickle.load(open('../../data_conala/conala_labelled_dataset_test.pkl', 'rb'))
    
    codes_train, asts_train, annos_train, annos2_train = zip(*train_dataset)
    codes_test, asts_test, annos_test, annos2_test = zip(*test_dataset)
    codes = pickle.load(open('../../data_conala/codes','rb'))
    annos = pickle.load(open('../../data_conala/annos','rb'))
    asts = pickle.load(open('../../data_conala/asts','rb'))
    if cfg["model"] == 'ct':
        train_dataset = list(zip(codes_train, annos_train, annos2_train))
        test_dataset = list(zip(codes_test, annos_test, annos2_test))
    


trainDF = {}
trainDF['code'] = codes
trainDF['anno'] = annos
trainDF['ast'] = asts

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
model_type = cfg["model"]
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
    device_id = cfg["device_id"]
    torch.cuda.set_device(device_id)

if cfg["dataset"]=='conala':
    save_path = save_path+"_conala"
elif cfg["dataset"]=='codesearchnet':
    pass
else:
    print("Wrong Dataset Entered")
    exit()

print(f"Number of epochs = {num_epochs}")
print(f"Batch size = {batch_size}")
print(f"Dataset = {cfg['dataset']}")
print(f"Model = {model_type.upper()}")

# Loading word embeddings

def prepare_sequence(seq, seq_len, to_ix):
    idxs_list = []
    
    for seq_elem in seq:
        idxs = []
        for w in seq_elem.split():
            try:
                idxs.append(to_ix[w])
            except KeyError:
                continue
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
seq_len_code = seq_len_anno = seq_len_ast = 300
load_var = False

if cfg["dataset"] == 'conala':
    word_to_ix_anno, weights_matrix_anno = create_embeddings(f'../../data_conala/anno_model.vec', 'anno')
    word_to_ix_code, weights_matrix_code = create_embeddings(f'../../data_conala/code_model.vec', 'code')
    word_to_ix_ast, weights_matrix_ast = create_embeddings(f'../../data_conala/ast_model.vec', 'ast')
    weights_matrix_anno = torch.from_numpy(weights_matrix_anno)
    weights_matrix_code = torch.from_numpy(weights_matrix_code)
    weights_matrix_ast = torch.from_numpy(weights_matrix_ast)
elif cfg["dataset"] == 'codesearchnet':
    if not load_var:
        word_to_ix_anno, weights_matrix_anno = create_embeddings(f'../{save_path}/anno_model.vec', 'anno')
        word_to_ix_code, weights_matrix_code = create_embeddings(f'../{save_path}/code_model.vec', 'code')
        word_to_ix_ast, weights_matrix_ast = create_embeddings(f'../{save_path}/ast_model.vec', 'ast')
        weights_matrix_anno = torch.from_numpy(weights_matrix_anno)
        weights_matrix_code = torch.from_numpy(weights_matrix_code)
        weights_matrix_ast = torch.from_numpy(weights_matrix_ast)
    else:
        word_to_ix_anno, weights_matrix_anno = pickle.load(open("../variables/anno_var",'rb'))
        word_to_ix_code, weights_matrix_code = pickle.load(open("../variables/code_var",'rb'))
        word_to_ix_ast, weights_matrix_ast = pickle.load(open("../variables/ast_var",'rb'))

if model_type == 'mpctm':
    sim_model = MPCTMClassifier(batch_size, weights_matrix_anno, weights_matrix_code, weights_matrix_ast, hidden_size, 
        num_layers_lstm, dense_dim, output_dim, seq_len_code).cuda()
elif model_type == 'ct':
    sim_model = CTModel(weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code)
elif model_type == 'cat':
    sim_model = CATModel(weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code, 
        weights_matrix_ast)

if torch.cuda.is_available() and use_cuda:
    sim_model.cuda()
    if use_parallel:
        sim_model = nn.DataParallel(sim_model)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

if use_softmax_classifier:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()

if use_adam:
    opt = torch.optim.Adam(sim_model.parameters(), lr=learning_rate)
else:
    opt = torch.optim.SGD(sim_model.parameters(), lr=learning_rate, momentum=0.9)
    
# Training
iter = 0
sim_model.train()
loss_file = open(f"losses/loss_file_{model_type}.csv",'w')
start_time = time.time()
if model_type == 'ct':
    for epoch in trange(num_epochs):
        epoch += 1
        batch_iter = 0
        loss_epoch = 0.0
        for i, (code_sequence, anno_sequence, anno_sequence_neg) in enumerate(tqdm(train_loader)):
            sim_model.zero_grad()
            anno_in = prepare_sequence(anno_sequence, seq_len_anno, word_to_ix_anno)
            code_in = prepare_sequence(code_sequence, seq_len_code, word_to_ix_code)
            anno_in_neg = prepare_sequence(anno_sequence_neg, seq_len_anno, word_to_ix_anno)
            if torch.cuda.is_available() and use_cuda:
                sim_score, _, _ = sim_model(anno_in.cuda(), code_in.cuda())
                sim_score_neg, _, _ = sim_model(anno_in_neg.cuda(), code_in.cuda())
            else:
                sim_score, _, _ = sim_model(anno_in, code_in)
                sim_score_neg, _, _ = sim_model(anno_in_neg, code_in)
            
            loss =  0.05 - sim_score + sim_score_neg
            loss[loss<0] = 0.0
            loss = torch.sum(loss)
            loss.backward()
            opt.step()
            iter += 1
            batch_iter += 1
            loss_epoch += loss
        loss_epoch /= batch_iter
        tqdm.write(f"Epoch: {epoch}. Loss: {loss_epoch}. Model: {model_type}")
        loss_file.write(f"{epoch},{loss_epoch}\n")
        if (epoch) % cfg["save_every"] == 0:
            torch.save(sim_model.state_dict(), f"../{save_path}/sim_model_cat_{(epoch)}")

elif model_type == 'cat':
    for epoch in trange(num_epochs):
        epoch += 1
        batch_iter = 0
        loss_epoch = 0.0
        for i, (code_sequence, ast_sequence, anno_sequence, anno_sequence_neg) in enumerate(tqdm(train_loader)):
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
            loss_epoch += loss
        loss_epoch /= batch_iter
        tqdm.write(f"Epoch: {epoch}. Loss: {loss_epoch}. Model: {model_type}")
        loss_file.write(f"{epoch},{loss_epoch}\n")
        if (epoch) % cfg["save_every"] == 0:
            torch.save(sim_model.state_dict(), f"../{save_path}/sim_model_cat_{(epoch)}")

elif model_type == 'mpctm':
    for epoch in trange(num_epochs):
        epoch += 1
        batch_iter = 0
        loss_epoch = 0.0
        for i, (code_sequence, ast_sequence, anno_sequence, anno_sequence_neg) in enumerate(tqdm(train_loader)):
            sim_model.zero_grad()
            anno_in = prepare_sequence(anno_sequence, seq_len_anno, word_to_ix_anno)
            code_in = prepare_sequence(code_sequence, seq_len_code, word_to_ix_code)
            ast_in = prepare_sequence(ast_sequence, seq_len_code, word_to_ix_ast)
            anno_in_neg = prepare_sequence(anno_sequence_neg, seq_len_anno, word_to_ix_anno)

            if torch.cuda.is_available() and use_cuda:
                sim_score = sim_model(anno_in.cuda(), code_in.cuda(), ast_in.cuda())
                sim_score_neg = sim_model(anno_in_neg.cuda(), code_in.cuda(), ast_in.cuda())
            else:
                sim_score = sim_model(anno_in, code_in, ast_in)
                sim_score_neg = sim_model(anno_in_neg, code_in, ast_in)
            
            loss =  0.05 - sim_score + sim_score_neg
            loss[loss<0] = 0.0
            loss = torch.sum(loss)
            loss.backward()
            opt.step()
            
            del anno_in, code_in, ast_in, anno_in_neg, sim_score, sim_score_neg
            
            iter += 1
            batch_iter += 1
            loss_epoch += loss
        loss_epoch /= batch_iter
        tqdm.write(f"Epoch: {epoch}. Loss: {loss_epoch}. Model: {model_type}")
        loss_file.write(f"{epoch},{loss_epoch}\n")
        if (epoch) % cfg["save_every"] == 0:
            torch.save(sim_model.state_dict(), f"../{save_path}/sim_model_mpctm_{(epoch)}")

print('Time taken to train: {} seconds'.format(time.time()-start_time))
loss_file.close()
print("Saving Models")
if model_type == 'mpctm':
    torch.save(sim_model.state_dict(), f"../{save_path}/sim_model_mpctm_reduced")
elif model_type == 'ct':
    torch.save(sim_model.state_dict(), f"../{save_path}/sim_model_ct")
elif model_type == 'cat':
    torch.save(sim_model.state_dict(), f"../{save_path}/sim_model_cat")
print("Saved Models")
