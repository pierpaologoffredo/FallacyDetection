import bisect
import re
import os
import pandas as pd
from Levenshtein import distance
import ipdb
from sklearn.model_selection import train_test_split

# def read_conll_file(conll_path):
#     i = 0
#     with open(conll_path, "r", encoding="utf-8") as f:
#         comp, rel = [], []
#         for line in f:
#             line = line.strip()
#             if line:
#                 line_strip = line.split("\t")
#                 if line_strip[0] == '0':
#                     rel.append(line_strip[2])
#                     comp.append(line_strip[3])
                    
#     return comp, rel
    
# def convert_to_df(tok, tag):
#     data_df = pd.DataFrame({'comp': tok, 'rel': tag})
#     return data_df


# ## Define the path of the annotation folder
# folder = './final_data/'

# ## Define the path of the annotation
# ## The data is already split in train, dev and test
# train_ann = os.path.join(folder, "train.conll")
# dev_ann = os.path.join(folder, "dev.conll")
# test_ann = os.path.join(folder, "test.conll")

# ## 1. Converting annotation in list of tokens and tags
# tr_w, tr_lab = read_conll_file(dev_ann)

# tr_df = convert_to_df(tr_w, tr_lab)
# print(tr_df['comp'].value_counts())
# # dev_data = read_conll_file(dev_ann)
# # test_data = read_conll_file(test_ann)

# def remove_duplicates_conll(file_path):
#     # Leggi il file CoNLL nel DataFrame utilizzando pandas
#     df = pd.read_csv(file_path, sep='\t', header=None, comment='#', skip_blank_lines=True)
#     print("BEFORE ", df.shape)
#     # Rimuovi i duplicati basandoti su tutte le colonne
#     df = df.drop_duplicates()
#     print("AFTER ", df.shape)
#     # # Salva il DataFrame senza duplicati in un nuovo file
#     # output_file_path = file_path.replace('.conll', '_noduplicates.conll')
#     # df.to_csv(output_file_path, sep='\t', header=None, index=False)

# # Esempio di utilizzo
# file_path = 'file.conll'
# remove_duplicates_conll(test_ann)


def tokenize_with_offsets(dial, fallacy):
    """ returns token, start and end offset of main text
        returns start and end offset of subtext considering the offsets of the main text
    """
    # ipdb.set_trace()
    d_tok, d_starts, d_ends = zip(*[(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', dial)])
    f_tok, f_starts, f_ends = zip(*[(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', fallacy)])

    # print(f_tok, f_starts, f_ends)

    dial_dict = {}
    for t, s, e in zip(d_tok, d_starts, d_ends):
        dial_dict[(s,e)]=re.sub(r'[^\w\s]', '', t)

    d_list = list(d_tok)
    f_list = list(f_tok)
    # print(d_list)
    # print(f_list)
    ner = []
    #if is_Sublist(d_list, f_list):
    #if re.search(r'\b{}'.format(fallacy if fallacy[-1]!='.' else fallacy[:-1]), dial):
    for i in range(len(f_list)):
        if i == 0:
            #start = ([k for k, v in dial_dict.items() if re.sub(r'[^\w\s]', '', v) == re.sub(r'[^\w\s]', '', f_list[i])][0][0])
            start = ([k for k, v in dial_dict.items() if (distance(re.sub(r'[^\w\s]', '', f_list[i]), re.sub(r'[^\w\s]', '', v)) <= 3 )][0][0])
        if i == len(f_list)-1:
            # print(i, f_list[i], f_list.index(f_list[i]))
            end = ([k for k, v in dial_dict.items() if re.sub(r'[^\w\s]', '', v) == re.sub(r'[^\w\s]', '', f_list[i])][0][1])        
            #x = [k for k, v in dial_dict.items() if (distance(re.sub(r'[^\w\s]', '', v), re.sub(r'[^\w\s]', '', f_list[i])) <= 3)]
            #end = ([k for k, v in dial_dict.items() if (distance(re.sub(r'[^\w\s]', '', v), re.sub(r'[^\w\s]', '', f_list[i])) <= 3)][0][1])        
    return d_tok, d_starts, d_ends, [(start, end)]

def get_labels(dial, fall, label):
    """ Convert offsets to sequence labels in BIO format."""
    d_t, d_s, d_e, span = tokenize_with_offsets(dial, fall)
    labels = ["O"]*len(d_s)
    spans = sorted(span)
    for s, e in spans:
        li = bisect.bisect_left(d_s, s)
        ri = bisect.bisect_left(d_s, e)
        ni = len(labels[li:ri])
        labels[li] = f"B-{label}"
        labels[li+1:ri] = [f"I-{label}"]*(ni-1)
    return labels

def operation(row):
    return get_labels(row['Context'], row['Fallacy'], row['Label'])

def clean_text(row):
    row['Context'] = row['Context'].apply(lambda x: re.sub(r'([a-z]+)(\-)([A-Za-z]+)', r'\1 \2 \3', str(x)))
    row['Context'] = row['Context'].apply(lambda x: re.sub(r'([a-z]+\.)([A-Za-z]+)', r'\1 \2', x))
    row['Context'] = row['Context'].apply(lambda x: re.sub(r'([a-z]+\?)([A-Za-z]+)', r'\1 \2', x))
    
# gold_df = pd.read_csv("all_fallacy_and_components_1960_2020.csv")
# gold_df.drop(columns={'PreviousSentence', 'CurrentSentence', 'NextSentence'}, inplace=True)
# gold_df['Context'] = gold_df['Context'].apply(lambda x: re.sub(r'([a-z]+)(\-)([A-Za-z]+)', r'\1 \2 \3', str(x)))
# gold_df['Context'] = gold_df['Context'].apply(lambda x: re.sub(r'([a-z]+\.)([A-Za-z]+)', r'\1 \2', x))
# gold_df['Context'] = gold_df['Context'].apply(lambda x: re.sub(r'([a-z]+\?)([A-Za-z]+)', r'\1 \2', x))
# gold_df['Fallacy'] = gold_df['Fallacy'].apply(lambda x: re.sub(r'([a-z]+)(\-)([A-Za-z]+)', r'\1 \2 \3', str(x)))
# gold_df['Fallacy'] = gold_df['Fallacy'].apply(lambda x: re.sub(r'([a-z]+\.)([A-Za-z]+)', r'\1 \2', x))
# gold_df['Fallacy'] = gold_df['Fallacy'].apply(lambda x: re.sub(r'([a-z]+\?)([A-Za-z]+)', r'\1 \2', x))
# gold_df = gold_df.drop_duplicates()
# gold_df['tags'] = gold_df.apply(operation, axis=1)
# train_set, test_set = train_test_split(gold_df, stratify=gold_df['Label'], test_size=0.1, random_state=42)

# train_set.to_csv("train_set.csv", index=False)
# test_set.to_csv("test_set.csv", index=False)

# # Reader of CONLL file
# def read_conll_file(conll_path):
#     num_sent = 0
#     sentences = []
#     with open(conll_path, "r", encoding="utf-8") as f:
#         words, labels, arg_comp, arg_rel = [], [], [], []
#         for line in f:
#             line = line.strip()
#             if not line:
#                 sentences.append((words, labels))
#                 words, labels, arg_comp, arg_rel = [], [], [], []
#             else:
#                 splits = line.split("\t")
#                 if splits[0] == '0':
#                     num_sent += 1
#                 words.append(splits[1])
#                 # arg_rel.append(splits[2])
#                 # arg_comp.append(splits[3])
#                 labels.append(splits[-1])   
#     print(num_sent)
#     return sentences

# # Processing the data into DataFrame
# def convert_bio_to_df(data):
    
#     data_df = pd.DataFrame(data, columns = ['tokens', 'tags'])
#     data_df['tokens'] = data_df['tokens'].apply(lambda x: " ".join(x))
#     data_df['tags'] = data_df['tags'].apply(lambda x: ", ".join(x))
#     return data_df

# # Extracting the fallacy label
# def extract_fallacy(lst):
#     for i in range(len(lst)):
#         if lst[i].startswith("B-"):
#             return lst[i][2:]
#     return None

# train_ann = read_conll_file("./test_data/train.conll")
# train_df = convert_bio_to_df(train_ann)
# test_ann = read_conll_file("./test_data/test.conll")
# test_df = convert_bio_to_df(test_ann)
# dev_ann = read_conll_file("./test_data/dev.conll")
# dev_df = convert_bio_to_df(dev_ann)
# merged = [train_df, test_df, dev_df]
# gold = pd.concat(merged)
# gold = gold.drop_duplicates().reset_index(drop=True)
# gold['labels'] = gold['tags'].apply(lambda x: extract_fallacy(x.split(", ")))

# test_set = pd.read_csv("test_set.csv")
# train_set = pd.read_csv("train_set.csv")
# merged_feat = [train_set, test_set]
# feat_gold = pd.concat(merged_feat).reset_index(drop=True)
# ipdb.set_trace()


# for i, fall in gold.iterrows():
#     for j, feat in feat_gold.iterrows():
#         if feat['Fallacy'] in fall['tokens']:
#             gold.loc[i, 'arg_comp'] = feat['CompLabel']
#             gold.loc[i, 'arg_rel'] = feat['RelLabel']
#             gold.loc[i, 'year'] = feat['FileName'][:4]

# ipdb.set_trace()

