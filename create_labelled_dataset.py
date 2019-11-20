import random
import pickle
import json
from tqdm import tqdm

def read_data(dataset_split='train', total_examples = 100000):
    # django_anno = open('data/all.anno')
    # django_code = open('data/all.code')
    # django_ast = open('data/all.ast')
    data = json.load(open(f'data/data_{dataset_split}.json'))

    count=0
    django_dataset=[]

    for elem in data:
        code = ' '.join(elem['code'])
        anno = ' '.join(elem['anno'])
        ast = ' '.join(elem['ast'])
        django_dataset.append((code, ast, anno, count))
        count+=1

    # for line_anno, line_code, line_ast in zip(django_anno, django_code, django_ast):
    #     django_dataset.append((line_code, line_ast, line_anno, count))
    #     count+=1

    # django_anno.close()
    # django_code.close()
    # django_ast.close()

    labelled_dataset=[]
    print(f"Reading {dataset_split} Set")
    for code, ast, anno, count in tqdm(django_dataset):
        count2 = count
        if dataset_split == 'train':
            while count==count2:
                code2, ast2, anno2, count2 = random.choice(django_dataset)
            labelled_dataset.append((code, ast, anno, anno2))
        else:
            distractor_list = []
            count_list = []
            for num_iter in range(999):
                while count==count2 or count2 in count_list:
                    code2, ast2, anno2, count2 = random.choice(django_dataset)
                distractor_list.append((code2, ast2))
                count_list.append(count2)
            labelled_dataset.append((code, ast, anno, distractor_list))

    return labelled_dataset

    count = 0
    for i in range(total_examples):
        count += 1
        print("Positive Example {}".format(count))
        django_code, django_ast, django_anno, ind = random.choice(django_dataset)
        while ind2==ind1:
            django_code2, django_ast2, django_anno2, ind2 = random.choice(django_dataset)
        labelled_dataset.append((django_code, django_ast, django_anno, django_anno2))
        labelled_dataset.append((django_code, django_ast, django_anno,1))

    # count = 0
    # for i in range(total_examples):
    #     count += 1
    #     print("Negative Example {}".format(count))
    #     django_code1, django_ast1, django_anno1, ind1 = random.choice(django_dataset)
    #     ind2=ind1
    #     while ind2==ind1:
    #         django_code2, django_ast2, django_anno2, ind2 = random.choice(django_dataset)
    #     labelled_dataset.append((django_code1, django_ast1, django_anno2, 0))

    random.shuffle(labelled_dataset)
    # print(labelled_dataset[:20])
    return labelled_dataset


def write_data(dataset, dataset_split='train'):
    # outp=open('data/django_labelled.tsv','w')
    # for i,j,k in dataset:
    #   outp.write(i+'\t'+j+'\t'+str(k)+'\n')
    pickle.dump( dataset, open( f"data/labelled_dataset_{dataset_split}.p", "wb" ) )


# dataset = read_data('train', 50000)
# write_data(dataset, 'train')

dataset = read_data('valid', 5000)
write_data(dataset, 'valid')

dataset = read_data('test', 5000)
write_data(dataset, 'test')
