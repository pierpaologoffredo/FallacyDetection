import os
import torch
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from utils import load_data, set_seed, start_wandb
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

      ## The tensor features are passed to the model
      outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
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
        
        ## The tensor features are passed to the model
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
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

def run_experiment(model_name, num_epochs, lr, device, cased=False, roberta=False, ignore_mismatched_sizes=False):  
    ########## RUNNING EXPERIMENT ##########

    ## Define the path of the annotation folder
    folder = './data/'

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

    #ipdb.set_trace()

    ## Set model configuration
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(labels_to_ids)
    config.id2label = ids_to_labels
    config.label2id = labels_to_ids

    ## Load ModelForTokenClassification
    if ignore_mismatched_sizes:
        model = AutoModelForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    
    model.num_labels = len(labels_to_ids)

    ## Move model to GPU
    model = model.to(device)

    best_val_loss = float('inf')
    best_epoch = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs)):
        
        print(f"\nTraining epoch: {epoch + 1}")
        ## TRAINING MODE
        train(epoch, model, training_loader, optimizer, device)

        ## EVALUATION MODE
        labels, predictions, curr_val_loss = valid(model, testing_loader, device, ids_to_labels)

        ## Saving the best model
        model_path = model_name + "_epochs_" + str(num_epochs)  
        output_model_dir = os.path.join("models/results", model_path)
        os.makedirs(output_model_dir, exist_ok=True)
        if curr_val_loss < best_val_loss:
            model.save_pretrained(output_model_dir)
            best_val_loss = curr_val_loss
            best_epoch = epoch
        ## Log metrics to wandb
        wandb.log({"epoch": epoch})

        ## Replacing BIO tags with normal tags for readable classification report
        full_predictions = [e.replace("B-", "").replace("I-", "") for e in predictions]
        full_labels = [e.replace("B-", "").replace("I-", "") for e in labels]

        ## Print the temporary classification report
        print("#"*100)
        print(classification_report(full_labels, full_predictions))
        print("#"*100)

    print()
    print(model_name)
    print("The {} model has been trained for {} epochs.".format(model_name, num_epochs))
    print("The model performed at its best at the {} epoch with validation loss of {}.".format(best_epoch, best_val_loss))
    """
    with open(os.path.join(output_model_dir, "report.txt"), "w") as f:
        f.write("######### All labels ##############\n")
        f.write(classification_report(labels, predictions))
        f.write("######### Simplified labels ##############\n")
        f.write(classification_report(labels2, predictions2))
    """

if __name__ == "__main__":
    
    
    # Selecting available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed for reproducibility
    torch.manual_seed(42)
    seed = 42
    set_seed(seed_num=seed)

    # Setting hyperparameters for training
    model_name = "bert-base-uncased"
    epochs = 15
    lr = 2e-05
    name = model_name + str(epochs)
    
    # Starting monitoring with wandb
    start_wandb(name, model_name, lr, epochs)
    
    run_experiment(model_name, epochs, lr, device=device)
    
    wandb.finish()