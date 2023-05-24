import os
import torch
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig, DistilBertForTokenClassification, BertForTokenClassification, AlbertForTokenClassification, RobertaForTokenClassification, ElectraForTokenClassification, LongformerForTokenClassification, DebertaForTokenClassification, XLNetForTokenClassification, XLMRobertaForTokenClassification
from utils import load_data, set_seed, start_wandb
import pandas as pd
from model.roberta import newRobertaForTokenClassification

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
      rels = batch['rels'].to(device, dtype = torch.long)
      rels = batch['rels'].to(device, dtype = torch.long)

      ## The tensor features are passed to the model
      outputs = model(input_ids=ids, attention_mask=mask, labels=labels, arg_comps=comps, arg_rels=rels)
      # outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
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
  wandb.log({"tr_loss": epoch_loss, "tr_accuracy": tr_accuracy})
  
  print(f"\nTraining loss epoch: {epoch_loss}\n")
  print(f"Training accuracy epoch: {tr_accuracy}\n")

def valid(model, testing_loader, device, ids_to_labels):
    ########## EVALUATION MODE ##########
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    # The model is set in training mode
    model.eval()

    with torch.no_grad():
      for idx, batch in enumerate(testing_loader):
        ## The features of the batch is converted into PyTorch tensors and moved on the GPU
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)
        comps = batch['comps'].to(device, dtype = torch.long)
        rels = batch['rels'].to(device, dtype = torch.long)
        
        ## The tensor features are passed to the model
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels, arg_comps=comps, arg_rels=rels)
        # outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        eval_logits = outputs.logits

        ## Calculate the loss
        eval_loss += loss.item()

        nb_eval_steps += 1
        nb_eval_examples += labels.size(0)
                  
        ## Calculate the prediction
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        ## Create the mask for useful values in labels
        ## The resulting tensor is made of boolean values
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
    
        ## Get the only useful features using the masks (active_accuracy) for targets and predictions
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        ## Fill the list of predictions and targets evaluated for this batch
        eval_labels.extend(labels)
        eval_preds.extend(predictions)
        
        ## Calculate the temporary accuracy of the model for this batch
        tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        eval_accuracy += tmp_eval_accuracy
        
        if idx % 500 == 0:
          print(f"\nValidation loss per 500 training steps: {eval_loss/nb_eval_steps:.4f}\n")


    ## Translate tensor values using ids_to_labels dictionary
    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]

    ## Log metrics to wandb
    wandb.log({"eval_loss": eval_loss / nb_eval_steps, "eval_accuracy": eval_accuracy / nb_eval_steps})

    print(f"\nValidation Loss: {eval_loss / nb_eval_steps:.4f}\n")
    print(f"Validation Accuracy: {eval_accuracy / nb_eval_steps:.4f}\n")
    
    return labels, predictions, eval_loss / nb_eval_steps

def test(model, tokenizer, test_dataset, ids_to_labels):
    ########## EVALUATION MODE ##########
    ## The model is set in training mode
    model.eval()

    ## Inizialize lists to store predictions
    pred_lines, list_cases = [], []
    
    for i, r in test_dataset.iterrows():
        ## Get the fallacy snippet and its tags splitted in list[]
        snippet = r["tokens"].strip().split()
        snippet_tags = r.tags.split(", ")

        ## Tokenize the input 
        inputs = tokenizer(
            snippet,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        ## The inputs tokenized are moved on the GPU
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        
        ## The inputs tokenized are passed to the model
        outputs = model(input_ids = ids, attention_mask = mask)
        
        ## Calculate the prediction
        logits = outputs[0]
        active_logits = logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size*seq_len,) - predictions at the token level

        ## Convert the tokens' ids into tokens
        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [
            ids_to_labels[i] for i in flattened_predictions.cpu().numpy()
        ]
        
        wp_preds = list(
            zip(tokens, token_predictions)
        )  # list of tuples. Each tuple = (wordpiece, prediction)

        ## Detokenize snippet and predictions
        detokenized_snippet = []
        detokenized_preds = []
        for t, p in zip(tokens, token_predictions):
            if t.startswith("##"):
                detokenized_snippet[-1] += t[2:]
                if p != "O":
                    detokenized_preds[-1] = p
            else:
                detokenized_snippet.append(t)
                detokenized_preds.append(p)
        
        ## Rebuilding the offset mapping of the tokenized predictions
        prediction = []
        wp_preds_tmp = []
        for token_pred, mapping in zip(
            wp_preds, inputs["offset_mapping"].squeeze().tolist()
        ):
            # only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
                wp_preds_tmp.append((token_pred, token_pred[1]))
            else:
                wp_preds_tmp.append((token_pred, token_pred[1]))
                continue

        ## Building the triplets (true_tok, true_tag, pred_tag)
        tmp_pred_lines = []
        for tok, true, pred in zip(snippet, snippet_tags, prediction):
            tmp_pred_lines.append((i, tok, true ,pred))
        pred_lines.append(tmp_pred_lines)

        list_cases.append(r["tokens"].strip())
    return list_cases, pred_lines

def make_reports(folder_path, preds, best=None):
    true_tags, pred_tags = [], []
    report = open(os.path.join(folder_path, "_report.txt"), "w") 
    ##### ALL BIO LABELS
    ## Storing the true and predicted token labels
    for snippet in preds:
        for tok in snippet:
            true_tags.append(tok[2])
            pred_tags.append(tok[3])
    print("The number of true tags are {} and the predicted ones are {}.".format(len(true_tags), len(pred_tags)))
    
    ## Printing the classification report with all the labels
    report.write("*"*20 + " CLASSIFICATION REPORT ALL BIO LABELS " + "*"*20 + "\n")
    report.write(classification_report(true_tags, pred_tags))    
    
    ##### ALL LABELS NORMALIZED
    ## Removing BI tags
    norm_true_tags = [tag.replace("B-", "").replace("I-", "") for tag in true_tags]
    norm_pred_tags = [tag.replace("B-", "").replace("I-", "") for tag in pred_tags]
    
    ## Printing the classification report with all the labels
    report.write("*"*20 + " CLASSIFICATION REPORT ALL NORMALIZED LABELS " + "*"*20 + "\n")
    report.write(classification_report(norm_true_tags, norm_pred_tags))
    
    ##### BINARY LABELS (NOT FALLACY, FALLACY)
    ## Converting tags into 0 and 1
    bin_true_tags = [1 if tag != "O" else 0 for tag in true_tags]
    bin_pred_tags = [1 if tag != "O" else 0 for tag in pred_tags]
    
    ## Printing the classification report with binary labels
    report.write("*"*20 + " CLASSIFICATION REPORT BINARY LABELS " + "*"*20 + "\n")
    report.write(classification_report(bin_true_tags, bin_pred_tags, target_names=["Not fallacy", "Fallacy"]))
    report.close()
    
    ## Creating a unique csv file with all the classification reports
    bio_cr = classification_report(true_tags, pred_tags, output_dict=True)
    cr = classification_report(norm_true_tags, norm_pred_tags, output_dict=True)
    bin_cr = classification_report(bin_true_tags, bin_pred_tags, target_names=["Not fallacy", "Fallacy"], output_dict=True)
    
    bio_cr_df = pd.DataFrame(bio_cr).transpose()
    cr_df = pd.DataFrame(cr).transpose()
    bin_cr_df = pd.DataFrame(bin_cr).transpose()
    
    all_data = [bio_cr_df, cr_df, bin_cr_df]
    
    final_df = pd.concat(all_data)
    
    final_df.to_csv(os.path.join(folder_path, "_report.csv"), index=False)

def run_experiment(model_name, num_epochs, lr, device, cased=False, roberta=False, ignore_mismatched_sizes=False):  
    ########## RUNNING EXPERIMENT ##########

    ## Define the path of the annotation folder
    folder = './final_data/'

    ## Define the path of the annotation
    ## The data is already split in train, dev and test
    train_ann = os.path.join(folder, "train.conll")
    dev_ann = os.path.join(folder, "dev.conll")
    test_ann = os.path.join(folder, "test.conll")

    ## Load the tokenizer
    if roberta:
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, test_dataset, training_set, testing_set, training_loader, testing_loader, labels_to_ids, ids_to_labels = load_data(train_path = train_ann, dev_path = dev_ann, device = device, test_path = test_ann, tokenizer = tokenizer, cased = cased)

    ## Set model configuration
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(labels_to_ids)
    config.id2label = ids_to_labels
    config.label2id = labels_to_ids
    config_feat = AutoConfig.from_pretrained(model_name)
    config_feat.num_labels = 3
    
    
    config_feat = AutoConfig.from_pretrained(model_name)
    config_feat.num_labels = 3
    
    
    ## Load ModelForTokenClassification
    if ignore_mismatched_sizes:
        # model = AutoModelForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model = newRobertaForTokenClassification.from_pretrained(model_name, config=config, config_feat=config_feat, ignore_mismatched_sizes=True)
        # model = RobertaForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    
    model.num_labels = len(labels_to_ids)

    ## Move model to GPU
    model = model.to(device)

    best_val_f1 = float('-inf')
    best_epoch = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs)):
        
        print(f"\nTraining epoch: {epoch + 1}")
        ## TRAINING MODE
        train(epoch, model, training_loader, optimizer, device)

        ## EVALUATION MODE
        labels, predictions, curr_val_loss = valid(model, testing_loader, device, ids_to_labels)

        ## Replacing BIO tags with normal tags for readable classification report
        full_predictions = [e.replace("B-", "").replace("I-", "") for e in predictions]
        full_labels = [e.replace("B-", "").replace("I-", "") for e in labels]

        ## Print the temporary classification report
        print("#"*100)
        print(classification_report(full_labels, full_predictions))
        print("#"*100)
        
        cr = classification_report(full_labels, full_predictions, output_dict=True)
        macro_f1_score = cr["macro avg"]["f1-score"]
        
        ## Saving the best model
        model_path = "" + model_name + "_epochs_" + str(num_epochs)  
        output_model_dir = os.path.join("models/edit_results", model_path)
        os.makedirs(output_model_dir, exist_ok=True)
        if macro_f1_score > best_val_f1:
            model.save_pretrained(output_model_dir)
            best_val_f1 = macro_f1_score
            best_epoch = epoch
        
        ## Log metrics to wandb
        wandb.log({"lr": lr, "epoch": epoch, "macro_avg f1": cr["macro avg"]["f1-score"]})
        

    print()
    print(model_name)
    best = ""
    print("The {} model has been trained for {} epochs. ".format(model_name, num_epochs))
    best += "The {} model has been trained for {} epochs. ".format(model_name, num_epochs)
    print("The model performed at its best at the {} epoch with validation loss of {}.".format(best_epoch, best_val_f1))
    best += "The model performed at its best at the {} epoch with validation loss of {}.".format(best_epoch, best_val_f1)
    
    ########## TESTING PHASE ##########
    # folder_path = "models/results/new_Jean-Baptiste/roberta-large-ner-english_epochs_20/"
    # model = RobertaForTokenClassification.from_pretrained("/models/results/new_Jean-Baptiste/roberta-large-ner-english_epochs_20/")
    # new_mod = RobertaForTokenClassification.from_pretrained(folder_path, config = config, ignore_mismatched_sizes=True)
    
    if model_name == "distilbert-base-uncased":
        new_mod = DistilBertForTokenClassification.from_pretrained(output_model_dir, config = config)
    elif model_name == "microsoft/deberta-base": 
        new_mod = DebertaForTokenClassification.from_pretrained(output_model_dir, config = config)
    elif model_name == "Jean-Baptiste/roberta-large-ner-english": #roberta true, ignore = true
        new_mod = newRobertaForTokenClassification.from_pretrained(output_model_dir, config = config, config_feat=config_feat, ignore_mismatched_sizes=True)
    # elif model_name == "bhadresh-savani/electra-base-discriminator-finetuned-conll03-english":
    #     new_mod = ElectraForTokenClassification.from_pretrained(output_model_dir, config = config, ignore_mismatched_sizes=True)
    # elif model_name == "brad1141/Longformer-finetuned-norm":
    #     new_mod = LongformerForTokenClassification.from_pretrained(output_model_dir, config = config, ignore_mismatched_sizes=True)
    # elif model_name == "bert-base-uncased":
    #     new_mod = BertForTokenClassification.from_pretrained(output_model_dir, config = config)
    # elif model_name == "albert-base-v2":
    #     new_mod = AlbertForTokenClassification.from_pretrained(output_model_dir, config = config)
    # elif model_name == "bert-large-cased":
    #     new_mod = BertForTokenClassification.from_pretrained(output_model_dir, config = config)
    # elif model_name == "xlnet-large-cased":
    #     new_mod = XLNetForTokenClassification.from_pretrained(output_model_dir, config = config)
    # elif model_name == "albert-xxlarge-v2":
    #     new_mod = AlbertForTokenClassification.from_pretrained(output_model_dir, config = config)
    # elif model_name == "distilbert-base-cased":
    #     new_mod = DistilBertForTokenClassification.from_pretrained(output_model_dir, config = config)
    # elif model_name == "electra-large-discriminator":
    #     new_mod = ElectraForTokenClassification.from_pretrained(output_model_dir, config = config, ignore_mismatched_sizes=True)
    # elif model_name == "xlm-roberta-large":
    #     new_mod = XLMRobertaForTokenClassification.from_pretrained(output_model_dir, config = config)
    # elif model_name == "albert-xxlarge-v2":
    #     new_mod = AlbertForTokenClassification.from_pretrained(output_model_dir, config = config)
    # elif model_name == "distilbert-base-cased":
    #     new_mod = DistilBertForTokenClassification.from_pretrained(output_model_dir, config = config)
    # elif model_name == "dbmdz/electra-large-discriminator-finetuned-conll03-english":
    #     new_mod = ElectraForTokenClassification.from_pretrained(output_model_dir, config = config, ignore_mismatched_sizes=True)
    
    # output_model_dir = "./models/upd_results/Jean-Baptiste/roberta-large-ner-english_epochs_30"
    # new_mod = RobertaForTokenClassification.from_pretrained(output_model_dir, config = config, ignore_mismatched_sizes=True)
    
    # Move model to GPU
    new_mod = new_mod.to(device)
    snippets, preds = test(model = new_mod, tokenizer = tokenizer, 
                                test_dataset = test_dataset, ids_to_labels = ids_to_labels)
    
    make_reports(output_model_dir, preds)
    

if __name__ == "__main__":
    
    
    # Selecting available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed for reproducibility
    torch.manual_seed(42)
    seed = 42
    set_seed(seed_num=seed)

    # uncased_model_list = ["distilbert-base-uncased", "bert-base-uncased", "albert-base-v2", 
    #               "Jean-Baptiste/roberta-large-ner-english", "bhadresh-savani/electra-base-discriminator-finetuned-conll03-english",
    #               "brad1141/Longformer-finetuned-norm", "microsoft/deberta-base"]
    
    # cased_model_list = [ "xlm-roberta-large", "bert-large-cased", "xlnet-large-cased",
    #                     "albert-xxlarge-v2", "distilbert-base-cased",
    #                     "dbmdz/electra-large-discriminator-finetuned-conll03-english"]

    # # Setting hyperparameters for training
    # # model_name = "Jean-Baptiste/roberta-large-ner-english"
    
    # for model_name in uncased_model_list:
    #     epochs = 30
    #     lr = 4e-05
    #     name = model_name + str(epochs)
        
    #     # Starting monitoring with wandb
    #     start_wandb(name, model_name, lr, epochs)
        
    #     if model_name == model_name == "microsoft/deberta-base": #roberta true
    #         run_experiment(model_name, epochs, lr, device=device, roberta=True)
    #     elif model_name == "Jean-Baptiste/roberta-large-ner-english": #roberta true, ignore = true
    #         run_experiment(model_name, epochs, lr, device=device, roberta=True, ignore_mismatched_sizes=True)
    #     elif model_name == "bhadresh-savani/electra-base-discriminator-finetuned-conll03-english" or model_name == "brad1141/Longformer-finetuned-norm":
    #         run_experiment(model_name, epochs, lr, device=device, ignore_mismatched_sizes=True)
    #     else:
    #         run_experiment(model_name, epochs, lr, device=device)
        
    #     wandb.finish()
    
        
    # for model_name in cased_model_list:
    #     # Setting hyperparameters for training
    #     # model_name = "bert-base-uncased"
    #     epochs = 30
    #     lr = 4e-05
    #     name = model_name + str(epochs)
        
    #     # Starting monitoring with wandb
    #     start_wandb(name, model_name, lr, epochs)
        
    #     run_experiment(model_name, epochs, lr, device=device, cased=True, ignore_mismatched_sizes=True)
        
    #     wandb.finish()
    
    model_name = "Jean-Baptiste/roberta-large-ner-english"
    epochs = 30
    lr = 2e-05
    name = model_name + str(epochs)
    
    # Starting monitoring with wandb
    start_wandb(name+str(lr), model_name, lr, epochs)
    run_experiment(model_name, epochs, lr, device=device, roberta=True, ignore_mismatched_sizes=True)
    wandb.finish()
