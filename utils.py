import pandas as pd
import torch
import os
import numpy as np
import wandb
from torch.utils.data import Dataset, DataLoader

# Reader of CONLL file
def read_conll_file(conll_path):
    sentences = []
    with open(conll_path, "r", encoding="utf-8") as f:
        words, labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                sentences.append((words, labels))
                words, labels = [], []
            else:
                splits = line.split("\t")
                words.append(splits[1])
                labels.append(splits[-1])
    return sentences

# Defining the custom class to batch the dataset
class FallacyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids, device):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.device = device

    def get_tokens_and_labels(self, index):
        tokens = self.data.tokens[index].strip().split()
        labels = self.data.tags[index].split(", ")
        return tokens, labels

    def __getitem__(self, index):
        ## 1. Get tokens and tags list of the item at position index
        tokens, tags = self.get_tokens_and_labels(index)

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

        ## 3. Convert all tags into numbers w/ labels_to_ids
        labels = [self.labels_to_ids[label] for label in tags]
        
        ## 3.1 Create an empty array filled of "-100" with size max length 
        encoded_labels = torch.full((self.max_len,), fill_value=-100, dtype=torch.long)

        ### Set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                try:
                    encoded_labels[idx] = labels[i]
                except:
                    print(tokens)
                i += 1
        
        ## 4. Convert all the data into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = encoded_labels
        item = {key: val.to(self.device) for key, val in item.items()}
        return item

    def __len__(self):
        return self.len
    
# Processing the data into DataFrame
def convert_bio_to_df(data):
  data_df = pd.DataFrame(data, columns = ['tokens', 'tags'])
  data_df['tokens'] = data_df['tokens'].apply(lambda x: " ".join(x))
  data_df['tags'] = data_df['tags'].apply(lambda x: ", ".join(x))
  
  return data_df

def load_data(train_path, test_path, dev_path, tokenizer, device,
                max_len = 128, train_batch_size = 8, valid_batch_size = 4, 
                cased = False):
  
    ########## LOADING DATA ##########

    ## Labels definition
    labels = ['B-AdHominem', 'I-AdHominem', 'B-AppealtoAuthority', 'I-AppealtoAuthority', 'B-AppealtoEmotion', 'I-AppealtoEmotion',
                'B-FalseCause', 'I-FalseCause', 'B-Slipperyslope', 'I-Slipperyslope', 'B-Slogans', 'I-Slogans', 'O']

    ## Label configuration
    labels_to_ids = {k: v for v, k in enumerate(labels)}
    ids_to_labels = {v: k for v, k in enumerate(labels)}

    ## 1. Converting annotation in list of tokens and tags
    train_data = read_conll_file(train_path)
    dev_data = read_conll_file(dev_path)
    test_data = read_conll_file(test_path)

    ## 2. Converting all the data into DataFrames
    train_df = convert_bio_to_df(train_data)
    dev_df = convert_bio_to_df(dev_data)
    test_df = convert_bio_to_df(test_data)

    ### 2.1 Concatenating train_df and dev_df into a single DataFrame
    data_merged = [train_df, dev_df]
    train_df = pd.concat(data_merged).reset_index(drop=True)

    ### 2.2 Concatenating train_df and test_df into a single DataFrame
    gold_data = [train_df, test_df]
    gold_df = pd.concat(gold_data).reset_index(drop=True)

    print("FULL Dataset: {}".format(gold_df.shape))
    print("TRAIN Dataset: {}".format(train_df.shape))
    print("TEST Dataset: {}".format(test_df.shape))

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

    ## Converting tokens in lower if needed
    if not cased:
        train_df['tokens'] = train_df['tokens'].str.lower()
        test_df['tokens'] = test_df['tokens'].str.lower()

    ## 3. Converting the DataFrame in FallacyDataset object
    training_set = FallacyDataset(train_df, tokenizer, max_len, labels_to_ids, device)
    testing_set = FallacyDataset(test_df, tokenizer, max_len, labels_to_ids, device)

    ## 4. Converting FallacyDataset into DataLoader object
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return train_df, test_df, training_set, testing_set, training_loader, testing_loader, labels_to_ids, ids_to_labels

def set_seed(seed_num):
    # Setting the seed
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    
def start_wandb(run_name, model_name, lr, epochs):
    os.environ["WANDB_API_KEY"] = "fb5654a489ac69b6a0e67c9a5bc773817fbcfe0a"
    # Starting a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="fallacy-ner",
        
        # set the name of the run
        name = run_name,
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": model_name,
        "dataset": "Fallacy",
        "epochs": epochs,
        }
    )