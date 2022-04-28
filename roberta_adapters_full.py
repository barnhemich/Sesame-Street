from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
import numpy as np
from transformers import RobertaConfig
from transformers import RobertaAdapterModel
from transformers import TrainingArguments, AdapterTrainer
from datasets import load_metric
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
import json
from datasets import load_metric

# dataset_name = 'SciERC'
dataset_name = 'citation_intent'

# mode = "FT"
mode = "TAPT+FT"

HP_tuning = False

#############################################################################################################################################

if dataset_name == "SciERC":
    proper_dataset_name = 'nsusemiehl/SciERC'
    dataset = load_dataset(proper_dataset_name)
    adapter_name = "SciERC"
elif dataset_name == "citation_intent":
    proper_dataset_name = 'zapsdcn/citation_intent'
    dataset = load_dataset(proper_dataset_name)
    adapter_name = "citation_intent"

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# Tokenize the set for the transformer
def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

# Encode the input data
# NOTE: num_proc does not seem to work, for some reason it can't find the tokenizer

# We make the labels the same as the input as this is language learning 
def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

if mode == "TAPT+FT": 
    dataset_pretraining = dataset.map(encode_batch, batched=True, remove_columns=dataset['train'].column_names,)
    dataset_pretraining = dataset_pretraining.map(add_labels, batched=True)
    dataset_pretraining.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

labels = np.unique(np.array(dataset['train']['label']))
num_of_labels = labels.size

# encoding the labels
def encode_labels(dataset):
    for i in range(num_of_labels):
        if dataset['label'] == labels[i]:
            dataset['label'] = i
    return dataset

if dataset_name == 'citation_intent':
    dataset = dataset.map(encode_labels, remove_columns=["metadata"])
    id2label={ 0: "Background", 1: "CompareOrContrast", 2: "Extends", 3: "Future", 4: "Motivation", 5: "Uses",}
else:
    dataset = dataset.map(encode_labels)
    id2label = {0:'COMPARE', 1:'CONJUNCTION', 2:'EVALUATE-FOR', 3:'FEATURE-OF', 4:'HYPONYM-OF', 5:'PART-OF', 6:'USED-FOR'}

# Encode the input data
dataset_finetuning = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
dataset_finetuning = dataset_finetuning.rename_column("label", 'labels')
# Transform to pytorch tensors and only output the required columns
dataset_finetuning.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

def model_init_FT():
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=num_of_labels,
    )
    model = RobertaAdapterModel.from_pretrained(
        "roberta-base",
        config=config,
    )
    model.add_adapter(adapter_name)
    model.add_classification_head(
            adapter_name,
            num_labels=num_of_labels,
            id2label=id2label,
            overwrite_ok = True)
    # Activate the adapter
    model.train_adapter(adapter_name)    
    return model

def model_init_TAPT():
    config = RobertaConfig.from_pretrained(
        "roberta-base",
    )
    model = RobertaAdapterModel.from_pretrained(
        "roberta-base",
        config=config,
    )     
    model.add_adapter(adapter_name)
    model.add_masked_lm_head(adapter_name)     
    model.train_adapter(adapter_name)    
    return model

def model_init_TAPT_FT():
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=num_of_labels,
    )
    model = RobertaAdapterModel.from_pretrained(
        "roberta-base",
        config=config,
    )
    model.load_adapter(adapter_dir)
    model.add_classification_head(
            adapter_name,
            num_labels=num_of_labels,
            id2label=id2label,
            overwrite_ok = True)
    # Activate the adapter
    model.train_adapter(adapter_name)    
    return model

metric = load_metric('f1')

def compute_metric(EvalPrediction):
  
  logits, labels = EvalPrediction
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels, average= 'macro')

def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [4, 8, 16]),
    }

if mode == "FT":
    output_dir="./training_output/finetuning/No_Pretrain"
    model = model_init_FT()

    writer = SummaryWriter(log_dir= f'runs/{adapter_name}')
    writer = TensorBoardCallback(writer)

    training_args = TrainingArguments(
        learning_rate=0.0007800713063316231,
        num_train_epochs=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        logging_steps=100,
        output_dir=output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=True,
        evaluation_strategy = 'steps',
        save_steps = 100,
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_finetuning["train"],
        eval_dataset=dataset_finetuning["validation"],
        callbacks=[writer],
        compute_metrics = compute_metric,
        model_init=model_init_FT 
    )

    if not HP_tuning:
        trainer.train()
        f = open(f"{output_dir}/evaluations.txt", "a")
        f.write(adapter_name)
        f.write(json.dumps(trainer.evaluate(dataset_finetuning['test'])))
        f.write('\n')
        f.close()
        
        # model.save_pretrained(f"{adapter_name}")
        model.save_all_adapters(output_dir)
        trainer.remove_callback(writer)
    else:
        # print(default_compute_objective())
        best_run = trainer.hyperparameter_search(direction="maximize", n_trials=30, hp_space=my_hp_space)
        print(best_run)

    # BestRun(run_id='18', objective=0.8756189300076448, hyperparameters={'learning_rate': 0.0003441611710974754, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 4}) scierc
    # BestRun(run_id='7', objective=0.6820728291316526, hyperparameters={'learning_rate': 0.0007800713063316231, 'per_device_train_batch_size': 8, 'per_device_eval_batch_size': 4}) acl

if mode == "TAPT+FT":
    # TAPT
    adapter_name = f"TAPT_{adapter_name}"
    output_dir = "./training_output/pretraining/TAPT"
    adapter_dir = f"{output_dir}/{adapter_name}"
    model = model_init_TAPT()
    
    writer = SummaryWriter(log_dir= f'runs/{adapter_name}')
    writer = TensorBoardCallback(writer)

    training_args = TrainingArguments(
        learning_rate=1.0556970664355873e-05,
        num_train_epochs=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        logging_steps=10,
        output_dir=output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=True,
        evaluation_strategy = 'steps',
        # load_best_model_at_end = True,
        save_steps = 100,
        gradient_accumulation_steps = 9,
        warmup_ratio = 0.06,
        # load_best_model_at_end = True,
        weight_decay=0.01,
        adam_epsilon = 1e-6,

    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_pretraining["train"],
        eval_dataset=dataset_pretraining["validation"],
        data_collator=data_collator,  
        callbacks=[writer],
        model_init=model_init_TAPT,
    )

    if not HP_tuning:
        trainer.train()
        f = open(f"{output_dir}/evaluations.txt", "a")
        f.write(adapter_name)
        f.write(json.dumps(trainer.evaluate(dataset_pretraining['test'])))
        f.write('\n')
        f.close()
    
    # model.save_pretrained(f"{adapter_name}")
        model.save_all_adapters(output_dir)
        trainer.remove_callback(writer)
    else:
        # print(default_compute_objective())
        best_run = trainer.hyperparameter_search(direction="maximize", n_trials=20, hp_space=my_hp_space)
        print(best_run)
        quit()
        # BestRun(run_id='13', objective=15.077034950256348, hyperparameters={'learning_rate': 1.0031749212835109e-05, 'per_device_train_batch_size': 16, 'per_device_eval_batch_size': 16}) scierc
        # BestRun(run_id='14', objective=15.997611999511719, hyperparameters={'learning_rate': 1.0556970664355873e-05, 'per_device_train_batch_size': 16, 'per_device_eval_batch_size': 8}) acl

    # FT
    output_dir = "./training_output/finetuning/TAPT"
    model = model_init_TAPT_FT()
    
    writer = SummaryWriter(log_dir= f'runs/{adapter_name}')
    writer = TensorBoardCallback(writer)

    training_args = TrainingArguments(
        learning_rate=0.0008097124089662972,
        num_train_epochs=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=100,
        output_dir=output_dir,
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        evaluation_strategy = 'epoch',
        # load_best_model_at_end = True,
        save_steps = 100,
        # lr_scheduler_type = 'constant',
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_finetuning["train"],
        eval_dataset=dataset_finetuning["validation"],
        callbacks=[writer],
        compute_metrics = compute_metric,
        model_init=model_init_TAPT_FT,
    )
    
    if not HP_tuning:
        trainer.train()
        f = open(f"{output_dir}/evaluations.txt", "a")
        f.write(adapter_name)
        f.write(json.dumps(trainer.evaluate(dataset_finetuning['test'])))
        f.write('\n')
        f.close()
        
        # model.save_pretrained(f"{adapter_name}")
        model.save_all_adapters(output_dir)
        trainer.remove_callback(writer)
    else:
        # print(default_compute_objective())
        best_run = trainer.hyperparameter_search(direction="maximize", n_trials=30, hp_space=my_hp_space)
        print(best_run)

        # BestRun(run_id='17', objective=0.8724980411135858, hyperparameters={'learning_rate': 0.0005388069849121145, 'per_device_train_batch_size': 8, 'per_device_eval_batch_size': 4}) scierc
        # BestRun(run_id='26', objective=0.6848311546840958, hyperparameters={'learning_rate': 0.0008097124089662972, 'per_device_train_batch_size': 8, 'per_device_eval_batch_size': 8}) acl