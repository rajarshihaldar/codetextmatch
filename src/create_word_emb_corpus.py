import json
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
from io import BytesIO
from parse_ast import process_ast
import ast
import gzip
import os
from tqdm import tqdm
import pickle

def tokenize_code(s):
    # Add the following imports
    # from tokenize import tokenize
    # from io import BytesIO
    result = ''
    flag = True
    g = tokenize(BytesIO(s.encode('utf-8')).readline)  # tokenize the string
    for toknum, tokval, _, _, _ in g:
        if tokval == 'utf-8':
            pass
        else:
            result+=' '+tokval
    return result

# out_anno = open('data/all.anno','w')
# out_ast = open('data/all.ast','w')
# out_code = open('data/all.code','w')

path = "../../data_codesearchnet/python/python/final/jsonl/"

def not_comment(seq):
    try:
        if seq[0] == '#':
            return False
        return True
    except:
        return True

def json_from_file(dataset):
    new_path = path + dataset + '/'
    input_data = []
    for filename in os.listdir(new_path):
        with gzip.open(new_path+filename, "rb") as f:
            for line in f:
                line = line.decode('utf-8')
                json_data = json.loads(line)
                input_data.append(json_data)
    output_data = []
    # count = 0
    for json_data in tqdm(input_data):
        json_elem = {}
        code = json_data['code_tokens']
        code = list(filter(not_comment, code))
        anno = json_data['docstring_tokens']
        json_elem['anno'] = anno
        json_elem['code'] = code
        json_elem['raw_anno'] = json_data['docstring']
        json_elem['raw_code'] = json_data['code']
        
        try:
            tree_node = ast.parse(json_data['code'])
            code_ast = process_ast('Root',tree_node)
        except SyntaxError:
            continue
        except RecursionError:
            print(json_data['code'])
            print("\n")
            continue
        json_elem['ast'] = code_ast.split()
        output_data.append(json_elem)
        # count += 1
        # print(count)
    return output_data

def json_to_file(output_json, dataset, indent = 4):
    with open(f"../data/data_{dataset}.json",'w') as fp:
        json.dump(output_json, fp, indent = 4)

def json_dump_text(output_json):
    count = 0
    with open("../data/all.code","a+") as fp:
        for data in output_json:
            fp.write(' '.join(data["code"]))
            fp.write("\n")
    with open("../data/all.anno","a+") as fp:
        for data in output_json:
            fp.write(' '.join(data["anno"]))
            fp.write("\n")
    with open("../data/all.ast","a+") as fp:
        for data in output_json:
            try:
                fp.write(' '.join(data["ast"]))
                fp.write("\n")
            except UnicodeEncodeError:
                count += 1
                pass
    print(f"Fail cases = {count}")
        
datasets = ['train', 'valid', 'test']

train_set = json_from_file('train')
print("Completed Processing Train Set")
valid_set = json_from_file('valid')
print("Completed Processing Valid Set")
test_set = json_from_file('test')
print("Completed Processing Test Set")

codes = []
annos = []
asts = []
for elem in train_set:
    code = ' '.join(elem['code'])
    anno = ' '.join(elem['anno'])
    ast = ' '.join(elem['ast'])
    codes.append(code)
    annos.append(anno)
    asts.append(ast)

for elem in valid_set:
    code = ' '.join(elem['code'])
    anno = ' '.join(elem['anno'])
    ast = ' '.join(elem['ast'])
    codes.append(code)
    annos.append(anno)
    asts.append(ast)

for elem in test_set:
    code = ' '.join(elem['code'])
    anno = ' '.join(elem['anno'])
    ast = ' '.join(elem['ast'])
    codes.append(code)
    annos.append(anno)
    asts.append(ast)

pickle.dump(codes, open('../data/codes','wb'))
pickle.dump(annos, open('../data/annos','wb'))
pickle.dump(asts, open('../data/asts','wb'))
exit()

with open("../data/all.code","w") as fp1, open("../data/all.anno","w") as fp2, open("../data/all.ast","w") as fp3:
    pass


json_to_file(train_set, 'train')
json_to_file(valid_set, 'valid')
json_to_file(test_set, 'test')

json_dump_text(train_set)
json_dump_text(valid_set)
json_dump_text(test_set)

