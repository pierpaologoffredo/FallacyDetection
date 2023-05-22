import pandas as pd
import torch
import os
import numpy as np
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import ipdb
from model.roberta import newRobertaForTokenClassification
from transformers import AutoConfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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

# Defining the custom class to batch the dataset
class FallacyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids, device):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.device = device

    def get_tokens_and_labels_and_comp_and_rel(self, index):
        tokens = self.data.tokens[index].strip().split()
        labels = self.data.tags[index].split(", ")
        comps = self.data.arg_comp[index].split(", ")
        rels = self.data.arg_rel[index].split(", ")
        print(len(tokens), len(labels), len(comps), len(rels))
        return tokens, labels, comps, rels

    def __getitem__(self, index):
        ## 1. Get tokens, tags, comps and rels list of the item at position index
        tokens, tags, comps, rels = self.get_tokens_and_labels_and_comp_and_rel(index)
        
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
        comps = [1 if comp == "Claim" else 0 for comp in comps]
        
        ## 3.1 Create an empty array filled of "-100" with size max length 
        encoded_labels = torch.full((self.max_len,), fill_value=-100, dtype=torch.long)
        encoded_comps = torch.full((self.max_len,), fill_value=-100, dtype=torch.long)
        
        ### Set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                try:
                    encoded_labels[idx] = labels[i]
                    encoded_comps[idx] = comps[i]
                except:
                    pass
                i += 1
        
        ## 4. Convert all the data into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = encoded_labels
        item["comps"] = encoded_comps
        item = {key: val.to(self.device) for key, val in item.items()}
        return item

    def __len__(self):
        return self.len
    
# Processing the data into DataFrame
def convert_bio_to_df(data):
    
    data_df = pd.DataFrame(data, columns = ['tokens', 'tags', 'arg_rel', 'arg_comp'])
    data_df['tokens'] = data_df['tokens'].apply(lambda x: " ".join(x))
    data_df['tags'] = data_df['tags'].apply(lambda x: ", ".join(x))
    data_df['arg_rel'] = data_df['arg_rel'].apply(lambda x: ", ".join(x))
    data_df['arg_comp'] = data_df['arg_comp'].apply(lambda x: ", ".join(x))
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
        project="fallacy_upd",
        
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
    
def train(epoch, model, training_loader, optimizer, device):
    ########## TRAINING MODE ##########

    MAX_GRAD_NORM = 10
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []

    # The model is set in training mode
    model.train()

    for idx, batch in enumerate(training_loader):
        
        ## The features of the batch is converted into PyTorch tensors and moved on the GPU
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)
        comps = batch['comps'].to(device, dtype = torch.long)

        ## The tensor features are passed to the model
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels, arg_feat=comps)
        loss = outputs.loss
        tr_logits = outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        with torch.no_grad():
            ## Calculate the prediction
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            ## Create the mask for useful values in labels
            ## The resulting tensor is made of boolean values
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
            
            ## Get the only useful features using the masks (active_accuracy) for targets and predictions
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            ## Fill the list of predictions and targets evaluated for this batch
            tr_labels.extend(labels)
            tr_preds.extend(predictions)
            
            ## Calculate the temporary accuracy of the model for this batch
            tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            ## Clip gradient to MAX_GRAD_NORM before exploding
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )
            
            ## Backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if idx % 1000 == 0:
            print(f"Training loss per 1000 training steps: {tr_loss/nb_tr_steps:.4f}\n")

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps

    ## Log metrics to wandb
    #   wandb.log({"tr_loss": epoch_loss, "tr_accuracy": tr_accuracy})

    print(f"\nTraining loss epoch: {epoch_loss}\n")
    print(f"Training accuracy epoch: {tr_accuracy}\n")
   
    
if __name__ == "__main__":
    ## Define the path of the annotation folder
    folder = './new_data/'

    ## Define the path of the annotation
    ## The data is already split in train, dev and test
    train_ann = os.path.join(folder, "train.conll")
    dev_ann = os.path.join(folder, "dev.conll")
    test_ann = os.path.join(folder, "test.conll")

    
    ########## LOADING DATA ##########

    ## Labels definition
    labels = ['B-AdHominem', 'I-AdHominem', 'B-AppealtoAuthority', 'I-AppealtoAuthority', 'B-AppealtoEmotion', 'I-AppealtoEmotion',
                'B-FalseCause', 'I-FalseCause', 'B-Slipperyslope', 'I-Slipperyslope', 'B-Slogans', 'I-Slogans', 'O']

    ## Label configuration
    labels_to_ids = {k: v for v, k in enumerate(labels)}
    ids_to_labels = {v: k for v, k in enumerate(labels)}

    ## 1. Converting annotation in list of tokens and tags
    # train_data = read_conll_file(train_ann)
    # dev_data = read_conll_file(dev_ann)
    test_data = read_conll_file(test_ann)

    # ## 2. Converting all the data into DataFrames
    # train_df = convert_bio_to_df(train_data)
    # dev_df = convert_bio_to_df(dev_data)
    test_df = convert_bio_to_df(test_data)
    
    

    # ### 2.1 Concatenating train_df and dev_df into a single DataFrame
    # data_merged = [train_df, dev_df]
    # train_df = pd.concat(data_merged).reset_index(drop=True)

    # ### 2.2 Concatenating train_df and test_df into a single DataFrame
    # gold_data = [train_df, test_df]
    # gold_df = pd.concat(gold_data).reset_index(drop=True)

    # print("FULL Dataset: {}".format(gold_df.shape))
    # print("TRAIN Dataset: {}".format(train_df.shape))
    # print("TEST Dataset: {}".format(test_df.shape))

    # train_params = {
    #     'batch_size': 8,
    #     'shuffle': True,
    #     'num_workers': 0
    # }

    test_params = {
        "batch_size": 4,
        "shuffle": False,
        "num_workers": 0,
    }
    
    # # Selecting available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "Jean-Baptiste/roberta-large-ner-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)


    # ## 3. Converting the DataFrame in FallacyDataset object
    # training_set = FallacyDataset(train_df, tokenizer, 128, labels_to_ids, device)
    testing_set = FallacyDataset(test_df, tokenizer, 128, labels_to_ids, device)
    
    testing_loader = DataLoader(testing_set, **test_params)
    
    
    ## Labels definition
    labels = ['B-AdHominem', 'I-AdHominem', 'B-AppealtoAuthority', 'I-AppealtoAuthority', 'B-AppealtoEmotion', 'I-AppealtoEmotion',
                'B-FalseCause', 'I-FalseCause', 'B-Slipperyslope', 'I-Slipperyslope', 'B-Slogans', 'I-Slogans', 'O']

    ## Label configuration
    labels_to_ids = {k: v for v, k in enumerate(labels)}
    ids_to_labels = {v: k for v, k in enumerate(labels)}
    
    ## Set model configuration
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(labels_to_ids)
    config.id2label = ids_to_labels
    config.label2id = labels_to_ids
    
    model = newRobertaForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
    model.num_labels = len(labels_to_ids)

    ## Move model to GPU
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-05)
    
    for epoch in tqdm(range(3)):
        
        print(f"\nTraining epoch: {epoch + 1}")
        ## TRAINING MODE
        train(epoch, model, testing_loader, optimizer, device)
    