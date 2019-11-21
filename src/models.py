import torch
import torch.nn as nn
import torch.utils.data

import numpy as np
import yaml

with open("../config.yml", 'r') as config_file:
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)

hidden_size = cfg["hidden_size"]
dense_dim = cfg["dense_dim"]
output_dim = cfg["output_dim"]
num_layers_lstm = cfg["num_layers_lstm"]
use_cuda = cfg["use_cuda"]
use_bidirectional = cfg["use_bidirectional"]

if use_cuda:
    device_id = 0
    torch.cuda.set_device(device_id)

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class LSTMModel(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, dense_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        if use_bidirectional:
            self.num_layers = num_layers * 2
        else:
            self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=use_bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(dense_dim*(int(use_bidirectional)+1), dense_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(dense_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if torch.cuda.is_available() and use_cuda:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Initialize cell state
        if torch.cuda.is_available() and use_cuda:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        else:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, (hn, cn) = self.lstm(self.embedding(x), (h0, c0))
        # out = self.fc(out[:, -1, :])
        out, _ = torch.max(out, dim=1, keepdim=False, out=None)
        out = self.fc(out)

        return out

class CTModel(nn.Module):
    def __init__(self, weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code):
        super(CTModel, self).__init__()
        self.anno_model = LSTMModel(weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, output_dim)
        self.code_model = LSTMModel(weights_matrix_code, hidden_size, num_layers_lstm, dense_dim, output_dim)
        self.dist = nn.modules.distance.PairwiseDistance(p=1, eps=1e-10)

    def forward(self, anno_in, code_in):
        anno_vector = self.anno_model(anno_in)
        code_vector = self.code_model(code_in)
        sim_score = 1.0-self.dist(anno_vector, code_vector)
        return sim_score, anno_vector, code_vector

class CATModel(nn.Module):
    def __init__(self, weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, output_dim, weights_matrix_code,
        weights_matrix_ast):
        super(CATModel, self).__init__()
        self.anno_model = LSTMModel(weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, 2*output_dim)
        self.code_model = LSTMModel(weights_matrix_code, hidden_size, num_layers_lstm, dense_dim, output_dim)
        self.ast_model = LSTMModel(weights_matrix_ast, hidden_size, num_layers_lstm, dense_dim, output_dim)
        self.dist = nn.modules.distance.PairwiseDistance(p=1, eps=1e-10)

    def forward(self, anno_in, code_in, ast_in):
        anno_vector = self.anno_model(anno_in)
        code_vector = self.code_model(code_in)
        ast_vector = self.ast_model(ast_in)
        code_ast_vector = torch.cat((code_vector, ast_vector), dim = 1)
        sim_score = 1.0-self.dist(anno_vector, code_ast_vector)
        return sim_score, anno_vector, code_vector

class LSTMModelMulti(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, dense_dim, output_dim):
        super(LSTMModelMulti, self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        if use_bidirectional:
            self.num_layers = num_layers * 2
        else:
            self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=use_bidirectional)

    def forward(self, x):
        if torch.cuda.is_available() and use_cuda:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Initialize cell state
        if torch.cuda.is_available() and use_cuda:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        else:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, (hn, cn) = self.lstm(self.embedding(x), (h0, c0))
        # out = self.fc(out[:, -1, :])

        return out

class DenseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(0.2),
            # nn.ReLU(),
        )

    def forward(self, x):
        # x = torch.cat((x1, x2), dim=1)
        out = self.fc1(x)
        return out


class BiMPMAggregator(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, output_dim):
        super(BiMPMAggregator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.num_layers = self.num_layers * 2
        self.output_dim = output_dim
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(self.hidden_dim*4, self.hidden_dim*2) 
        self.fc2 = nn.Linear(self.hidden_dim*2, self.output_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        # Linear function 4 (readout): 100 --> 10
        # self.fc2 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x1, x2):
        if torch.cuda.is_available() and use_cuda:
            h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim).cuda()
            h1 = torch.zeros(self.num_layers, x2.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim)
            h1 = torch.zeros(self.num_layers, x2.size(0), self.hidden_dim)
        if torch.cuda.is_available() and use_cuda:
            c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim).cuda()
            c1 = torch.zeros(self.num_layers, x2.size(0), self.hidden_dim).cuda()
        else:
            c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim)
            c1 = torch.zeros(self.num_layers, x2.size(0), self.hidden_dim)
        out1, (hn0, cn0) = self.lstm1(x1, (h0, c0))
        out2, (hn1, cn1) = self.lstm2(x2, (h1, c1))
        out1 = out1[:, -1, :]
        out2 = out2[:, -1, :]
        
        # Linear function 1
        out = torch.cat((out1,out2), dim = 1)
        out = self.fc1(out)
        # Non-linearity 1
        out = self.relu1(out)
        # Linear function 2
        out = self.fc2(out)
        return out


class MPCTMAggregator(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, output_dim):
        super(MPCTMAggregator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.num_layers = self.num_layers * 2
        self.output_dim = output_dim
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(self.hidden_dim*4, self.hidden_dim*2) 
        self.fc2 = nn.Linear(self.hidden_dim*2, self.output_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        self.dist = nn.modules.distance.PairwiseDistance(p=2, eps=1e-10)
        # Linear function 4 (readout): 100 --> 10
        # self.fc2 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x1, x2, anno, code):
        if torch.cuda.is_available() and use_cuda:
            h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim).cuda()
            h1 = torch.zeros(self.num_layers, x2.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim)
            h1 = torch.zeros(self.num_layers, x2.size(0), self.hidden_dim)
        if torch.cuda.is_available() and use_cuda:
            c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim).cuda()
            c1 = torch.zeros(self.num_layers, x2.size(0), self.hidden_dim).cuda()
        else:
            c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim)
            c1 = torch.zeros(self.num_layers, x2.size(0), self.hidden_dim)
        out1, (hn0, cn0) = self.lstm1(x1, (h0, c0))
        out2, (hn1, cn1) = self.lstm2(x2, (h1, c1))
        out1 = out1[:, -1, :]
        out2 = out2[:, -1, :]

        out1 = torch.cat((out1, anno), dim=1)
        out2 = torch.cat((out2, code), dim=1)
        
        # # Linear function 1
        # out = torch.cat((out1,out2), dim = 1)
        # out = self.fc1(out)
        # # Non-linearity 1
        # out = self.relu1(out)
        # # Linear function 2
        # out = self.fc2(out)
        out1 = out1.norm(p=2, dim=1, keepdim=True)
        out2 = out2.norm(p=2, dim=1, keepdim=True)
        out = 1.0-self.dist(out1, out2)
        # out = cos(out1, out2)
        return out


class BiMPMLayer(nn.Module):
    def __init__(self, batch_size, weights_matrix_anno, weights_matrix_code, weights_matrix_ast, hidden_size, num_layers_lstm, dense_dim, output_dim, seq_len):
        super(BiMPMLayer, self).__init__()
        self.multi_anno_model = LSTMModelMulti(weights_matrix_anno, 2*hidden_size, num_layers_lstm, dense_dim, 
            2*output_dim)
        self.multi_code_model = LSTMModelMulti(weights_matrix_code, hidden_size, num_layers_lstm, dense_dim, output_dim)
        self.multi_ast_model = LSTMModelMulti(weights_matrix_ast, hidden_size, num_layers_lstm, dense_dim, output_dim)
        self.model= nn.ModuleList()
        for i in range(16):
            new_model = DenseModel(2*hidden_size, 2*hidden_size, 2*hidden_size)
            self.model.append(new_model)
        # self.aggregation_model = AggregationModel(12, 1, seq_len, 2)
        self.cos = nn.CosineSimilarity(dim=2)
        self.batch_size = batch_size


    def full_matching(self, anno_forward, code_forward, anno_reverse, code_reverse, model1, model2, model3, model4):
        # model1 = dense_model1
        # model2 = dense_model2
        # model3 = dense_model11
        # model4 = dense_model12
        
        
        # per_tensor = []
        # rev_per_tensor = []
        anno_vector_all = model1(anno_forward)
        anno_vector_all_rev = model2(anno_reverse)

        code_ast_vector = model3(code_forward[:,-1,:])
        # code_ast_vector = torch.nn.functional.normalize(code_ast_vector, p=2, dim=1, eps=1e-12, out=None)
        code_ast_vector_rev = model4(code_reverse[:,-1,:])
        # code_ast_vector_rev = torch.nn.functional.normalize(code_ast_vector_rev, p=2, dim=1, eps=1e-12, out=None)

        anno_vector_all = anno_vector_all.permute(1,0,2)
        code_ast_vector = code_ast_vector.view(1,self.batch_size,400)
        per_tensor = self.cos(code_ast_vector, anno_vector_all)
        per_tensor = per_tensor.permute(1,0)
        per_tensor = per_tensor.unsqueeze(2)

        anno_vector_all_rev = anno_vector_all_rev.permute(1,0,2)
        code_ast_vector_rev = code_ast_vector_rev.view(1,self.batch_size,400)
        rev_per_tensor = self.cos(code_ast_vector_rev, anno_vector_all_rev)
        rev_per_tensor = rev_per_tensor.permute(1,0)
        rev_per_tensor = rev_per_tensor.unsqueeze(2)
        
        code_vector_all = model3(code_forward)
        code_vector_all_rev =model4(code_reverse)
        
        anno_vector = model1(anno_forward[:,-1,:])
        # anno_vector = torch.nn.functional.normalize(anno_vector, p=2, dim=1, eps=1e-12, out=None)
        anno_vector_rev = model2(anno_reverse[:,-1,:])
        # anno_vector_rev = torch.nn.functional.normalize(anno_vector_rev, p=2, dim=1, eps=1e-12, out=None)
        # sim_score = 1.0-dist(anno_vector_rev, code_vector_all.permute(0,2,1))
        
        code_vector_all = code_vector_all.permute(1,0,2)
        anno_vector = anno_vector.view(1, self.batch_size,400)
        per_tensor2 = self.cos(anno_vector, code_vector_all)
        per_tensor2 = per_tensor2.permute(1,0)
        per_tensor2 = per_tensor2.unsqueeze(2)

        code_vector_all_rev = code_vector_all_rev.permute(1,0,2)
        anno_vector_rev = anno_vector_rev.view(1, self.batch_size, 400)
        rev_per_tensor2 = self.cos(anno_vector_rev, code_vector_all_rev)
        rev_per_tensor2 = rev_per_tensor2.permute(1,0)
        rev_per_tensor2 = rev_per_tensor2.unsqueeze(2)
        
        full_match_tensor1 = torch.cat((per_tensor, rev_per_tensor), dim = 2)
        full_match_tensor2 = torch.cat((per_tensor2, rev_per_tensor2), dim = 2)
        # full_match_tensor = torch.cat((full_match_tensor1,full_match_tensor2), dim = 2)

        return full_match_tensor1, full_match_tensor2


    def div_with_small_value(self, n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def maxpool_matching(self, anno_forward, code_forward, anno_reverse, code_reverse):

        anno_vector = self.model[4](anno_forward)
        code_ast_vector = self.model[5](code_forward)

        anno_norm = anno_vector.norm(p=2, dim=2, keepdim=True)
        code_norm = code_ast_vector.norm(p=2, dim=2, keepdim=True)

        d = anno_norm * code_norm.transpose(1, 2)

        n = torch.matmul(anno_vector, code_ast_vector.transpose(1,2))
        m = self.div_with_small_value(n, d)

        anno_vector = self.model[6](anno_reverse)
        code_ast_vector = self.model[7](code_reverse)

        anno_norm = anno_vector.norm(p=2, dim=2, keepdim=True)
        code_norm = code_ast_vector.norm(p=2, dim=2, keepdim=True)

        d = anno_norm * code_norm.transpose(1, 2)

        n = torch.matmul(anno_vector, code_ast_vector.transpose(1,2))
        m2 = self.div_with_small_value(n, d)

        return m,m2
        


    def attention(self, v1, v2):
        """
        :param v1: (batch, seq_len1, hidden_size)
        :param v2: (batch, seq_len2, hidden_size)
        :return: (batch, seq_len1, seq_len2)
        """

        # (batch, seq_len1, 1)
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        # (batch, 1, seq_len2)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

        # (batch, seq_len1, seq_len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))
        d = v1_norm * v2_norm

        return self.div_with_small_value(a, d)


    def attentive_matching(self, anno_forward, code_forward, anno_reverse, code_reverse):
        atf = self.model[8](anno_forward)
        atb = self.model[9](anno_reverse)
        ctf = self.model[10](code_forward)
        ctb = self.model[11](code_reverse)

        # atf = dense_model3(anno_forward)
        # atb = dense_model4(anno_reverse)
        # ctf = dense_model13(code_forward)
        # ctb = dense_model14(code_reverse)
        att_fw = self.attention(atf, ctf)
        att_bw = self.attention(atb, ctb)
        

        att_h_fw = ctf.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = ctb.unsqueeze(1) * att_bw.unsqueeze(3)

        att_p_fw = atf.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = atb.unsqueeze(2) * att_bw.unsqueeze(3)

        att_mean_h_fw = self.div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = self.div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

        att_mean_p_fw = self.div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = self.div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

        att1 = self.full_matching(atf, att_mean_h_fw, atb, att_mean_h_bw, self.model[8], self.model[9], 
            self.model[10], self.model[11])
        att1 = torch.cat((att1[0],att1[1]), dim=2)
        att2 = self.full_matching(ctf, att_mean_p_fw, ctb, att_mean_p_bw, self.model[8], self.model[9], 
            self.model[10], self.model[11])
        att2 = torch.cat((att2[0],att2[1]), dim=2)

        # attentive_tensor = torch.cat((att1, att2), dim=2)
        return att1, att2





    def max_attentive_matching(self, anno_forward, code_forward, anno_reverse, code_reverse):
        atf = self.model[12](anno_forward)
        atb = self.model[13](anno_reverse)
        ctf = self.model[14](code_forward)
        ctb = self.model[15](code_reverse)

        # atf = dense_model3(anno_forward)
        # atb = dense_model4(anno_reverse)
        # ctf = dense_model13(code_forward)
        # ctb = dense_model14(code_reverse)
        att_fw = self.attention(atf, ctf)
        att_bw = self.attention(atb, ctb)
        

        att_h_fw = ctf.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = ctb.unsqueeze(1) * att_bw.unsqueeze(3)

        att_p_fw = atf.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = atb.unsqueeze(2) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)


        max1 = self.full_matching(atf, att_max_h_fw, atb, att_max_h_bw, self.model[12], self.model[13], 
            self.model[14], self.model[15])

        max2 = self.full_matching(att_max_p_fw, ctf, att_max_p_bw, ctb, self.model[12], self.model[13], 
            self.model[14], self.model[15])

        max1 = torch.cat((max1[0],max1[1]), dim=2)
        max2 = torch.cat((max2[0],max2[1]), dim=2)
        # max_attentive_tensor = torch.cat((max1, max2), dim=2)
        return max1, max2

    def forward(self, anno_in, code_in, ast_in):
        self.batch_size = anno_in.size()[0]
        multi_anno_vector = self.multi_anno_model(anno_in)
        multi_code_vector = self.multi_code_model(code_in)
        multi_ast_vector = self.multi_ast_model(ast_in)
        multi_code_ast_vector = torch.cat((multi_code_vector, multi_ast_vector), dim = 2)
        anno_forward, anno_reverse = torch.split(multi_anno_vector,2*hidden_size, dim=2)
        code_forward,code_reverse = torch.split(multi_code_ast_vector,2*hidden_size, dim=2)
        full_match_tensor1, full_match_tensor2 = self.full_matching(anno_forward, code_forward, anno_reverse, 
            code_reverse, self.model[0], self.model[1], self.model[2], self.model[3])

        mv_max_fw, mv_max_bw = self.maxpool_matching(anno_forward, code_forward, anno_reverse, code_reverse)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)

        mv_p_max_fw = mv_p_max_fw.unsqueeze(2)
        mv_p_max_bw = mv_p_max_bw.unsqueeze(2)
        mv_h_max_fw = mv_h_max_fw.unsqueeze(2)
        mv_h_max_bw = mv_h_max_bw.unsqueeze(2)

        maxpool_match_tensor1 = torch.cat((mv_p_max_fw,mv_p_max_bw),dim=2)
        maxpool_match_tensor2 = torch.cat((mv_h_max_fw,mv_h_max_bw),dim=2)

        attentive_tensor1, attentive_tensor2 = self.attentive_matching(anno_forward, code_forward, anno_reverse, 
            code_reverse)

        max_attentive_tensor1, max_attentive_tensor2 = self.max_attentive_matching(anno_forward, code_forward, 
            anno_reverse, code_reverse)
        
        feature_tensor1 = torch.cat((full_match_tensor1, maxpool_match_tensor1, attentive_tensor1, max_attentive_tensor1)
            , dim = 2)

        feature_tensor2 = torch.cat((full_match_tensor2, maxpool_match_tensor2, attentive_tensor2, max_attentive_tensor2)
            , dim = 2)

        # outputs = self.aggregation_model(feature_tensor1, feature_tensor2)

        return feature_tensor1, feature_tensor2

class BiMPMClassifier(nn.Module):
    def __init__(self, batch_size, weights_matrix_anno, weights_matrix_code, weights_matrix_ast, hidden_size, num_layers_lstm, dense_dim, output_dim, seq_len):
        super(BiMPMClassifier, self).__init__()
        self.bimpm_layer = BiMPMLayer(batch_size, weights_matrix_anno, weights_matrix_code, weights_matrix_ast, hidden_size, num_layers_lstm, dense_dim, output_dim, seq_len)
        self.classifier_model = BiMPMAggregator(12, 1, seq_len, 2)

    def forward(self, anno_in, code_in, ast_in):
        feature_tensor1, feature_tensor2 = self.bimpm_layer(anno_in, code_in, ast_in)
        outputs = classifier_model(feature_tensor1, feature_tensor2)
        return outputs


class MPCTMClassifier(nn.Module):
    def __init__(self, batch_size, weights_matrix_anno, weights_matrix_code, weights_matrix_ast, hidden_size, num_layers_lstm, dense_dim, output_dim, seq_len):
        super(MPCTMClassifier, self).__init__()
        self.anno_model = LSTMModel(weights_matrix_anno, hidden_size, num_layers_lstm, dense_dim, 2*output_dim)
        self.code_model = LSTMModel(weights_matrix_code, hidden_size, num_layers_lstm, dense_dim, output_dim)
        self.ast_model = LSTMModel(weights_matrix_ast, hidden_size, num_layers_lstm, dense_dim, output_dim)
        self.bimpm_layer = BiMPMLayer(batch_size, weights_matrix_anno, weights_matrix_code, weights_matrix_ast, hidden_size, num_layers_lstm, dense_dim, output_dim, seq_len)
        self.classifier_model = MPCTMAggregator(12, 1, seq_len, 2)

    def forward(self, anno_in, code_in, ast_in):
        anno_vector = self.anno_model(anno_in)
        code_vector = self.code_model(code_in)
        ast_vector = self.ast_model(ast_in)
        code_ast_vector = torch.cat((code_vector, ast_vector), dim = 1)
        feature_tensor1, feature_tensor2 = self.bimpm_layer(anno_in, code_in, ast_in)
        sim_score = self.classifier_model(feature_tensor1, feature_tensor2, anno_vector, code_ast_vector)
        return sim_score


