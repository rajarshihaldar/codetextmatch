from sklearn import model_selection
import pandas
import pickle
import numpy as np
import time
import json
import yaml
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.utils.data
from operator import itemgetter
from models import MPCTMClassifier, CATModel, CTModel

with open("../config.yml", 'r') as config_file:
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)

if cfg["dataset"] == "codesearchnet":
    test_dataset = pickle.load(open('../data/labelled_dataset_test.p', 'rb'))
    codes = pickle.load(open('../data/codes','rb'))
    annos = pickle.load(open('../data/annos','rb'))
    asts = pickle.load(open('../data/asts','rb'))
elif cfg["dataset"] == 'conala':
    test_dataset = pickle.load(open('../../data_conala/conala_labelled_dataset_test.pkl', 'rb'))
    codes = pickle.load(open('../../data_conala/codes','rb'))
    annos = pickle.load(open('../../data_conala/annos','rb'))
    asts = pickle.load(open('../../data_conala/asts','rb'))


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
print(f"Model = {cfg['model']}")


# Loading word embeddings
if use_bin:
    import fastText.FastText as ft
    ft_anno_vec = ft.load_model('conala/ft_models/anno_model.bin')
    ft_code_vec = ft.load_model('conala/ft_models/code_model.bin')
else:
    from keras.preprocessing import text, sequence

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
load_var = False
seq_len_code = seq_len_anno = seq_len_ast = 300
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
        num_layers_lstm, dense_dim, output_dim, seq_len_code)
    if torch.cuda.is_available() and use_cuda:
        sim_model.cuda()
    sim_model.load_state_dict(torch.load(f"../{save_path}/sim_model_mpctm_reduced"))
elif model_type == 'ct':
    sim_model = CTModel(weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code)
    if torch.cuda.is_available() and use_cuda:
        sim_model.cuda()
    sim_model.load_state_dict(torch.load(f"../{save_path}/sim_model_ct"))
elif model_type == 'cat':
    sim_model = CATModel(weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code, 
        weights_matrix_ast)
    if torch.cuda.is_available() and use_cuda:
        sim_model.cuda()
    sim_model.load_state_dict(torch.load(f"../{save_path}/sim_model_cat"))


ret_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)

def eval_ct():
    mrr = 0
    count = 0
    r1 = 0
    r5 = 0
    r10 = 0

    rank_list = []
    sim_model.eval()
    with torch.no_grad():
        for i, (code_sequence, ast_sequence, anno_sequence, distractor_list) in enumerate(tqdm(ret_loader)):
            # print("Idx = ", i)
            anno_in = prepare_sequence(anno_sequence, seq_len_anno, word_to_ix_anno)
            ranked_list = []
            codebase = []
            codebase.append(code_sequence[0])
            count_dist = 0
            for code_dist, ast_dist in distractor_list:
                # count_dist += 1
                # if count_dist >= 99:
                #     break
                codebase.append(code_dist[0])
            if torch.cuda.is_available() and use_cuda:
                anno_in = anno_in.cuda()
            # anno_vector = anno_model(anno_in)
            for cand_code in codebase:
                cand_code = (cand_code,)
                code_in = prepare_sequence(cand_code, seq_len_code, word_to_ix_code)
                
                if torch.cuda.is_available() and use_cuda:
                    code_in = code_in.cuda()
                
                sim_score, _, _ = sim_model(anno_in, code_in)

                # code_vector = code_model(code_in)
                # if torch.cuda.is_available() and use_cuda:
                #     labels = labels.cuda()
                # cos = nn.CosineSimilarity(dim=1, eps=1e-10)
                # dist = nn.modules.distance.PairwiseDistance(p=1, eps=1e-10)
                # sim_score = 1.0-dist(anno_vector, code_vector)
                
                sim_score = sim_score.item()
                ranked_list.append((cand_code, sim_score))
            ranked_list = sorted(ranked_list, key=itemgetter(1),reverse=True)
            # ranked_list = ranked_list[:11]
            code_sequence = code_sequence[0]
            rank = 0
            for i, item in enumerate(ranked_list):
                if item[0][0]==code_sequence and rank==0:
                    rank = i+1
            if not rank:
                count += 1
                continue
                # print("No Rank")
                # exit()
            rank_list.append([anno_sequence, code_sequence, rank])
            mrr += 1.0/(rank)
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
                # outp.write("Rank = {}\n".format(rank))
                # outp.write(' '.join(anno_sequence[0]))
                # outp.write("\n")
                for i in range(5):
                    code = ranked_list[i][0]
                #     outp.write(' '.join(code[0]))
                #     outp.write("\n")
                # outp.write("\n")
            if rank <= 10:
                r10 += 1
            count += 1

    # outp.close()
    mrr /= count
    r1 /= count
    r5 /= count
    r10 /= count
    df = pandas.DataFrame(rank_list, columns=['Query','Gold', 'Rank'])
    df.to_pickle(f"../results/results_CT_{cfg['dataset']}.pkl")
    with open(f"../results/results_CT_{cfg['dataset']}.txt","w") as f:
        f.write(f"MRR = {mrr}\n")
        f.write(f"Recall@1 = {r1}\n")
        f.write(f"Recall@5 = {r5}\n")
        f.write(f"Recall@10 = {r10}\n")
    print("MRR = ", mrr)
    print("Recall@1 = ",r1)
    print("Recall@5 = ",r5)
    print("Recall@10 = ",r10)

def eval_cat():
    mrr = 0
    count = 0
    r1 = 0
    r5 = 0
    r10 = 0

    rank_list = []
    sim_model.eval()
    with torch.no_grad():
        for i, (code_sequence, ast_sequence, anno_sequence, distractor_list) in enumerate(tqdm(ret_loader)):
            # print("Idx = ", i)
            anno_in = prepare_sequence(anno_sequence, seq_len_anno, word_to_ix_anno)
            ranked_list = []
            codebase = []
            codebase.append((code_sequence[0], ast_sequence[0]))
            count_dist = 0
            for code_dist, ast_dist in distractor_list:
                count_dist += 1
                # if count_dist >= 99:
                #     break
                codebase.append((code_dist[0], ast_dist[0]))
            if torch.cuda.is_available() and use_cuda:
                anno_in = anno_in.cuda()
            # anno_vector = anno_model(anno_in)
            for cand_code, cand_ast in codebase:
                cand_code = (cand_code,)
                cand_ast = (cand_ast,)
                code_in = prepare_sequence(cand_code, seq_len_code, word_to_ix_code)
                ast_in = prepare_sequence(cand_ast, seq_len_ast, word_to_ix_ast)
                
                if torch.cuda.is_available() and use_cuda:
                    code_in = code_in.cuda()
                    ast_in = ast_in.cuda()
                
                sim_score, _, _ = sim_model(anno_in, code_in, ast_in)
                sim_score = sim_score.item()
                ranked_list.append((cand_code, sim_score))
            ranked_list = sorted(ranked_list, key=itemgetter(1),reverse=True)
            # ranked_list = ranked_list[:11]
            code_sequence = code_sequence[0]
            rank = 0
            for i, item in enumerate(ranked_list):
                if item[0][0]==code_sequence and rank==0:
                    rank = i+1
            if not rank:
                count += 1
                continue
                # print("No Rank")
                # exit()
            rank_list.append([anno_sequence, code_sequence, rank])
            mrr += 1.0/(rank)
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
                # outp.write("Rank = {}\n".format(rank))
                # outp.write(' '.join(anno_sequence[0]))
                # outp.write("\n")
                for i in range(5):
                    code = ranked_list[i][0]
                    # outp.write(' '.join(code[0]))
                    # outp.write("\n")
                # outp.write("\n")
            if rank <= 10:
                r10 += 1
            count += 1

    # outp.close()
    mrr /= count
    r1 /= count
    r5 /= count
    r10 /= count
    df = pandas.DataFrame(rank_list, columns=['Query','Gold', 'Rank'])
    df.to_pickle(f"../results/results_CAT_{cfg['dataset']}.pkl")
    with open(f"../results/results_CAT_{cfg['dataset']}.txt","w") as f:
        f.write(f"MRR = {mrr}\n")
        f.write(f"Recall@1 = {r1}\n")
        f.write(f"Recall@5 = {r5}\n")
        f.write(f"Recall@10 = {r10}\n")
    print("MRR = ", mrr)
    print("Recall@1 = ",r1)
    print("Recall@5 = ",r5)
    print("Recall@10 = ",r10)

def eval_mpctm():
    mrr = 0
    count = 0
    r1 = 0
    r5 = 0
    r10 = 0

    rank_list = []
    sim_model.eval()
    with torch.no_grad():
        for i, (code_sequence, ast_sequence, anno_sequence, distractor_list) in enumerate(tqdm(ret_loader)):
            # print("Idx = ", i)
            anno_in = prepare_sequence(anno_sequence, seq_len_anno, word_to_ix_anno)
            ranked_list = []
            codebase = []
            codebase.append((code_sequence[0], ast_sequence[0]))
            count_dist = 0
            for code_dist, ast_dist in distractor_list:
                count_dist += 1
                # if count_dist >= 99:
                #     break
                codebase.append((code_dist[0], ast_dist[0]))
            if torch.cuda.is_available() and use_cuda:
                anno_in = anno_in.cuda()
            # anno_vector = anno_model(anno_in)
            for cand_code, cand_ast in codebase:
                cand_code = (cand_code,)
                cand_ast = (cand_ast,)
                code_in = prepare_sequence(cand_code, seq_len_code, word_to_ix_code)
                ast_in = prepare_sequence(cand_ast, seq_len_ast, word_to_ix_ast)
                
                if torch.cuda.is_available() and use_cuda:
                    code_in = code_in.cuda()
                    ast_in = ast_in.cuda()
                
                sim_score = sim_model(anno_in, code_in, ast_in)

                # code_vector = code_model(code_in)
                # if torch.cuda.is_available() and use_cuda:
                #     labels = labels.cuda()
                # cos = nn.CosineSimilarity(dim=1, eps=1e-10)
                # dist = nn.modules.distance.PairwiseDistance(p=1, eps=1e-10)
                # sim_score = 1.0-dist(anno_vector, code_vector)
                
                sim_score = sim_score.item()
                ranked_list.append((cand_code, sim_score))
            ranked_list = sorted(ranked_list, key=itemgetter(1),reverse=True)
            # ranked_list = ranked_list[:11]
            code_sequence = code_sequence[0]
            rank = 0
            for i, item in enumerate(ranked_list):
                if item[0][0]==code_sequence and rank==0:
                    rank = i+1
            if not rank:
                count += 1
                continue
                # print("No Rank")
                # exit()
            rank_list.append([anno_sequence, code_sequence, rank])
            mrr += 1.0/(rank)
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
                # outp.write("Rank = {}\n".format(rank))
                # outp.write(' '.join(anno_sequence[0]))
                # outp.write("\n")
                for i in range(5):
                    code = ranked_list[i][0]
                    # outp.write(' '.join(code[0]))
                    # outp.write("\n")
                # outp.write("\n")
            if rank <= 10:
                r10 += 1
            count += 1

    # outp.close()
    mrr /= count
    r1 /= count
    r5 /= count
    r10 /= count
    df = pandas.DataFrame(rank_list, columns=['Query','Gold', 'Rank'])
    df.to_pickle(f"../results/results_MPCTM_{cfg['dataset']}.pkl")
    with open(f"../results/results_MPTCTM_{cfg['dataset']}.txt","w") as f:
        f.write(f"MRR = {mrr}\n")
        f.write(f"Recall@1 = {r1}\n")
        f.write(f"Recall@5 = {r5}\n")
        f.write(f"Recall@10 = {r10}\n")
    print("MRR = ", mrr)
    print("Recall@1 = ",r1)
    print("Recall@5 = ",r5)
    print("Recall@10 = ",r10)

if model_type == 'ct':
    eval_ct()
elif model_type == 'cat':
    eval_cat()
elif model_type == 'mpctm':
    eval_mpctm()