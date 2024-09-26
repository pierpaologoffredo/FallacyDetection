import pandas as pd
import torch
import os
import numpy as np
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import ipdb
from transformers import AutoConfig
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import spacy 
import ast

# Reader of CONLL file
def read_conll_file(conll_path):
    sentences = []
    with open(conll_path, "r", encoding="utf-8") as f:
        words, labels, arg_comp, arg_rel = [], [], [], []
        for line in f:
            line = line.strip()
            if not line:
                sentences.append((words, labels, arg_rel, arg_comp))
                words, labels, arg_comp, arg_rel = [], [], [], []
            else:
                splits = line.split("\t")
                words.append(splits[1])
                arg_rel.append(splits[2])
                arg_comp.append(splits[3])
                labels.append(splits[-1])
    return sentences

# Creating the set of PoS tags
def get_unique_pos_tags(pos_strings):
    return sorted(list({tag for pos_string in pos_strings for tag in pos_string.split(",")}))

# Defining the custom class to batch the dataset
class FallacyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids, pos_tags_to_ids, device):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.device = device
        self.pos_tags_to_ids = pos_tags_to_ids

    def get_tokens_and_labels_and_comp_and_rel(self, index):
        # ipdb.set_trace()
        
        tokens = self.data.Context[index].strip().split()
        labels = ast.literal_eval(self.data.bio_tags[index])
        comps = [self.data.arg_comp[index]] * len(labels)
        rels = [self.data.arg_rel[index]] * len(labels)
        pos_tags = self.data.pos_tags[index].split(",")
        print(len(tokens), len(labels), len(comps), len(rels))
        return tokens, labels, comps, rels, pos_tags

    def __getitem__(self, index):
        ## 1. Get tokens, tags, comps and rels list of the item at position index
        tokens, tags, comps, rels, pos_tags = self.get_tokens_and_labels_and_comp_and_rel(index)
        # tokens, tags, comps, rels = self.get_tokens_and_labels_and_comp_and_rel(index)
        
        ## 2: Tokenize tokens including padding/truncation up to max_length
        ### BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        

        ## 3. Convert all features into numbers w/ labels_to_ids
        # ipdb.set_trace()
        labels = [self.labels_to_ids[label] for label in tags]
        comps = [1 if comp == "Claim" else 2 if comp == "Premise" else 0 for comp in comps]
        rels = [1 if rel == "Attack" else 2 if rel == "Support" else 0 for rel in rels]

        ## 3.1 Create the dictionary of PoS tags occured into the DataFrame
        pos_tags = [self.pos_tags_to_ids[pos] for pos in pos_tags]
        
        ## 4. Create an empty array filled of "-100" with size max length 
        encoded_labels = torch.full((self.max_len,), fill_value=-100, dtype=torch.long)
        encoded_comps = torch.full((self.max_len,), fill_value=-100, dtype=torch.long)
        encoded_rels = torch.full((self.max_len,), fill_value=-100, dtype=torch.long)
        encoded_pos_tags = torch.full((self.max_len,), fill_value=-100, dtype=torch.long)
        
        ## 4.1. Filling the encoded_pos_tags
        for i in range(len(pos_tags)):
            encoded_pos_tags[i] = pos_tags[i]

        ### Set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                try:
                    encoded_labels[idx] = labels[i]
                    encoded_comps[idx] = comps[i]
                    encoded_rels[idx] = rels[i]
                    encoded_rels[idx] = rels[i]
                except:
                    pass
                i += 1
        
        ## 5. Convert all the data into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = encoded_labels
        item["comps"] = encoded_comps
        item["rels"] = encoded_rels
        item["pos_tags"] = encoded_pos_tags
        item = {key: val.to(self.device) for key, val in item.items()}
        return item

    def __len__(self):
        return self.len

def load_data(tokenizer, device, frac_data = 1, max_len = 128, train_batch_size = 8, valid_batch_size = 4, cased = False):
  
    ########## LOADING DATA ##########

    ## Labels definition
    labels = ['B-AdHominem', 'I-AdHominem', 'B-AppealtoAuthority', 'I-AppealtoAuthority', 'B-AppealtoEmotion', 'I-AppealtoEmotion',
                'B-FalseCause', 'I-FalseCause', 'B-Slipperyslope', 'I-Slipperyslope', 'B-Slogans', 'I-Slogans', 'O']

    ## Label configuration
    labels_to_ids = {k: v for v, k in enumerate(labels)}
    ids_to_labels = {v: k for v, k in enumerate(labels)}

    ## 1. Reading train and test set
    train_df = pd.read_csv('./data/pos_train_set.csv')
    test_df = pd.read_csv('./data/pos_test_set.csv')
    
    ## 2. Evaluating the amount of data taken into consideration
    if frac_data < 1:
        old_train_df, train_df = train_test_split(train_df, stratify=train_df['Label'], test_size=frac_data, random_state=42)
        old_test_df, test_df = train_test_split(test_df, stratify=test_df['Label'], test_size=frac_data, random_state=42)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True) 

    # print("FULL Dataset: {}".format(gold_df.shape))
    print("TRAIN Dataset: {}".format(train_df.shape))
    print("TEST Dataset: {}".format(test_df.shape))
    print(train_df['Label'].value_counts(normalize=True))
    print(test_df['Label'].value_counts(normalize=True))
    
    if not cased:
        train_df['Context'] = train_df['Context'].str.lower()
        test_df['Context'] = test_df['Context'].str.lower()

    ### 2.1 Extracting the set of PoS tags 
    pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
                'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    
    ## 2.2 PoS tags configuration
    pos_tags_to_ids = {k: v for v, k in enumerate(pos_tags)}

    train_params = {
        'batch_size': train_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    test_params = {
        "batch_size": valid_batch_size,
        "shuffle": False,
        "num_workers": 0,
    }

    ## 3. Converting the DataFrame in FallacyDataset object
    training_set = FallacyDataset(train_df, tokenizer, max_len, labels_to_ids, pos_tags_to_ids, device)
    testing_set = FallacyDataset(test_df, tokenizer, max_len, labels_to_ids, pos_tags_to_ids, device)

    ## 4. Converting FallacyDataset into DataLoader object
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return train_df, test_df, training_set, testing_set, training_loader, testing_loader, labels_to_ids, ids_to_labels

def set_seed(seed_num):
    # Setting the seed
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)