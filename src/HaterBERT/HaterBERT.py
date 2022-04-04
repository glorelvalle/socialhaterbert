import os
import torch
import gc
import glob
import random
import time
import neptune
import datetime
import transformers
import pandas as pd
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

from multilingual_bert import *

from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, KFold

from transformers.models.bert.modeling_bert import *
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer


# Check GPU
device = torch.device("cuda")
print("The is %d GPU(s) availables." % torch.cuda.device_count())
print("Using GPU:", torch.cuda.get_device_name(0))

# Set the gpu device
print("Current GPU device", torch.cuda.current_device())
torch.cuda.set_device(0)

api_token = "YOUR_KEY_NEPTUNE"
project_name = "YOUR_DIR_NEPTUNE"

batch_size = 8
MAX_LEN = 512


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def custom_tokenize(sentences, tokenizer, max_length=512):
    """This function tokenize given sentences"""
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        try:
            encoded_sent = tokenizer.encode(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_length,
                # This function also supports truncation and conversion
                # to pytorch tensors, but we need to do padding, so we
                # can't use these features :( .
                # max_length = 128,          # Truncate all sentences.
                # return_tensors = 'pt',     # Return pytorch tensors.
            )

        # Add the encoded sentence to the list.
        except ValueError:
            encoded_sent = tokenizer.encode(
                " ",  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_length,
                # This function also supports truncation and conversion
                # to pytorch tensors, but we need to do padding, so we
                # can't use these features :( .
                # max_length = 128,          # Truncate all sentences.
                # return_tensors = 'pt',     # Return pytorch tensors.
            )
        input_ids.append(encoded_sent)

    return input_ids


def custom_attention_masks(input_ids):
    """Creates attention mask for the given inputs"""
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return attention_masks


def combine_preprocessing(sentences, tokenizer, max_length=512):
    """Truncate and Tokenize sentences, then pad them"""
    input_ids = custom_tokenize(sentences, tokenizer, max_length)
    input_ids = pad_sequences(
        input_ids, dtype="long", value=0, truncating="post", padding="post"
    )
    att_masks = custom_attention_masks(input_ids)
    return input_ids, att_masks


def generate_dataloader(input_ids, labels, att_masks, batch_size=8, is_train=False):
    """Generate PyTorch data loader with the given dataset"""
    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels, dtype=torch.long)
    masks = torch.tensor(np.array(att_masks))
    data = TensorDataset(inputs, masks, labels)
    if is_train == False:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def stratified_sample_df(df, col, n_samples, sampled="stratified", random_state=1):
    """Sample the given dataframe df to select n_sample number of points"""
    if sampled == "stratified":
        df_ = (
            df.groupby(col, group_keys=False)
            .apply(lambda x: x.sample(int(np.rint(n_samples * len(x) / len(df)))))
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    elif sampled == "equal":
        df_ = (
            df.groupby(col, group_keys=False)
            .apply(lambda x: x.sample(int(n_samples / 2)))
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    return df_


def data_collector(file_names, params, is_train):
    """Data collection taking all at a time"""

    if params["csv_file"] == "*_full.csv":
        index = 12
    elif params["csv_file"] == "*_translated.csv":
        index = 23
    sample_ratio = params["sample_ratio"]
    type_train = params["how_train"]
    sampled = params["samp_strategy"]
    take_ratio = params["take_ratio"]
    language = params["language"]

    print("Language {0}".format(language))

    # If the data being loaded is not train, i.e. either val or test, load everything and return
    if is_train != True:
        df_test = []
        for file in file_names:
            lang_temp = file.split("/")[-1][:-index]
            if lang_temp == language:
                df_test.append(pd.read_csv(file))
        df_test = pd.concat(df_test, axis=0)
        return df_test

    # If train data is being loaded,
    else:
        # Baseline setting - only target language data is loaded
        if type_train == "baseline":
            df_test = []
            for file in file_names:

                lang_temp = file.split("/")[-1][:-index]
                if lang_temp == language:
                    temp = pd.read_csv(file)
                    df_test.append(temp)
            df_test = pd.concat(df_test, axis=0)

        # Zero shot setting - all except target language loaded
        if type_train == "zero_shot":
            df_test = []
            for file in file_names:
                lang_temp = file.split("/")[-1][:-index]
                if lang_temp == "English":
                    temp = pd.read_csv(file)

                    df_test.append(temp)
            df_test = pd.concat(df_test, axis=0)

        # All_but_one - all other languages fully loaded, target language sampled
        if type_train == "all_but_one":
            df_test = []
            for file in file_names:
                lang_temp = file.split("/")[-1][:-index]
                if lang_temp != language:
                    temp = pd.read_csv(file)
                    df_test.append(temp)
            df_test = pd.concat(df_test, axis=0)

        if take_ratio == True:
            n_samples = int(len(df_test) * sample_ratio / 100)
        else:
            # n_samples=sample_ratio
            n_samples = int(len(df_test))

        if n_samples == 0:
            n_samples += 1
        df_test = stratified_sample_df(
            df_test, "label", n_samples, sampled, params["random_seed"]
        )
        return df_test


def fix_the_random(seed_val=42):
    """Function to set the random seeds for reproducibility"""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    """Function to calculate the accuracy of our predictions vs labels"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_fscore(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average="macro")


def save_model(model, tokenizer, params):
    """Function to save models"""

    if params["to_save"] == True:
        if params["csv_file"] == "*_full.csv":
            translate = "translated"
        else:
            translate = "actual"

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if params["how_train"] != "all":
            output_dir = (
                "models_saved/"
                + params["path_files"]
                + "_"
                + params["language"]
                + "_"
                + translate
                + "_"
                + params["how_train"]
                + "_"
                + str(params["sample_ratio"])
            )
        else:
            output_dir = (
                "models_saved/"
                + params["path_files"]
                + "_"
                + translate
                + "_"
                + params["how_train"]
                + "_"
                + str(params["sample_ratio"])
            )

        if params["save_only_bert"]:
            model = model.bert
            output_dir = output_dir + "_only_bert/"
        else:
            output_dir = output_dir + "/"

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model
        # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


# Function to select model based on parameters passed
def select_model(type_of_model, path, weights=None, label_list=None):
    if type_of_model == "weighted":
        model = SC_weighted_BERT.from_pretrained(
            path,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            weights=weights,
        )
    elif type_of_model == "normal":
        model = BertForSequenceClassification.from_pretrained(
            path,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
    elif type_of_model == "multitask":
        model = BertForMultitask.from_pretrained(
            path,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            label_uniques=label_list,
        )
    else:
        print("Error in model name.")
    return model


class SC_weighted_BERT(BertPreTrainedModel):
    """Class for weighted bert for sentence classification"""

    def __init__(self, config, weights):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights = weights
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:

                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=torch.tensor(self.weights).cuda())
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForMultitask(BertPreTrainedModel):
    """BERT for multitask learning"""

    def __init__(self, config, label_uniques):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout_list = []
        self.classifier_list = []
        self.label_uniques = label_uniques
        for ele in self.label_uniques:
            self.dropout_list.append(nn.Dropout(config.hidden_dropout_prob))
            self.classifier_list.append(nn.Linear(config.hidden_size, ele))
        self.dropout_list = torch.nn.ModuleList(self.dropout_list)
        self.classifier_list = torch.nn.ModuleList(self.classifier_list)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        logits_list = []
        for i in range(len(self.label_uniques)):
            output_1 = self.dropout_list[i](pooled_output)
            logits = self.classifier_list[i](output_1)
            logits_list.append(logits)
        outputs = (logits_list,) + outputs[
            2:
        ]  # add hidden states and attention if they are here
        loss = 0
        for i in range(len(self.label_uniques)):
            # label=torch.nn.functional.one_hot(labels[:,i])
            label = labels[:, i]
            loss_fct = CrossEntropyLoss(reduction="mean").cuda()
            loss += loss_fct(
                logits_list[i].view(-1, self.label_uniques[i]), label.view(-1)
            )
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def eval_phase(params, which_files="test", model=None):
    """Evaluation phase"""

    # For english, there is no translation, hence use full dataset.
    if params["language"] == "English":
        params["csv_file"] = "*_full.csv"

    # Load the files to test on

    if which_files == "train":
        path = params["files"] + "/train/" + params["csv_file"]
        test_files = glob.glob(path)
    if which_files == "val":
        path = params["files"] + "/val/" + params["csv_file"]
        test_files = glob.glob(path)
    if which_files == "test":
        path = params["files"] + "/test/" + params["csv_file"]
        test_files = glob.glob(path)

    """Testing phase of the model"""
    print("Loading BERT tokenizer...")

    # Load bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(params["path_files"], do_lower_case=False)

    # If model is passed, then use the given model. Else load the model from the saved location
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    if params["is_model"] == True:
        print("model previously passed")
        model.eval()
    else:
        model = select_model(
            params["what_bert"], params["path_files"], params["weights"]
        )
        model.cuda()
        model.eval()

    # Load the dataset
    print("-- Load dataset --")
    print(test_files)

    df_test = data_collector(test_files, params, False)
    if params["csv_file"] == "*_translated.csv":
        sentences_test = df_test.translated.values
    elif params["csv_file"] == "*_full.csv":
        sentences_test = df_test.text.values

    labels_test = df_test.label.values
    # Encode the dataset using the tokenizer
    input_test_ids, att_masks_test = combine_preprocessing(
        sentences_test, tokenizer, params["max_length"]
    )
    test_dataloader = generate_dataloader(
        input_test_ids,
        labels_test,
        att_masks_test,
        batch_size=params["batch_size"],
        is_train=False,
    )
    print("Running eval on ", which_files, "...")
    t0 = time.time()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    true_labels = []
    pred_labels = []
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        # pred_labels+=list(np.argmax(logits, axis=0).flatten())
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())

        # Track the number of batches
        nb_eval_steps += 1

    # Get the accuracy and macro f1 scores
    testf1 = f1_score(true_labels, pred_labels, average="macro")
    testacc = accuracy_score(true_labels, pred_labels)
    try:
        testauc = roc_auc_score(true_labels, pred_labels, average="macro")
    except ValueError:
        testauc = 0
    testpred = precision_score(true_labels, pred_labels, average="macro")
    testrec = recall_score(true_labels, pred_labels, average="macro")
    cnf_matrix = confusion_matrix(true_labels, pred_labels)
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix,
        classes=["Non_hate", "Hate"],
        title="Confusion matrix, without normalization",
    )
    plt.show()

    # Log the metrics obtained
    if params["logging"] != "neptune" or params["is_model"] == True:
        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.5f}".format(testacc))
        print(" Fscore: {0:.5f}".format(testf1))
        print(" AUC: {0:.5f}".format(testauc))
        print(" Precision: {0:.5f}".format(testpred))
        print(" Recall: {0:.5f}".format(testrec))
        print(" Test took: {:}".format(format_time(time.time() - t0)))
    else:  # neptune
        bert_model = params["path_files"][:-1]
        language = params["language"]
        name_one = bert_model + "_" + language
        neptune.create_experiment(
            name_one,
            params=params,
            send_hardware_metrics=False,
            run_monitoring_thread=False,
        )
        neptune.append_tag(bert_model)
        neptune.append_tag(language)
        neptune.append_tag("test")
        neptune.log_metric("test_f1score", testf1)
        neptune.log_metric("test_accuracy", testacc)
        neptune.log_metric("test_auc", testauc)
        neptune.log_metric("test_precision", testpred)
        neptune.log_metric("test_recall", testrec)
        neptune.stop()

    return testf1, testacc, testauc, testpred, testrec


def train_model(params, best_val_fscore):
    """Main function that does the training"""

    # In case of english languages, translation is the origin data itself.
    if params["language"] == "English":
        params["csv_file"] = "*_full.csv"

    train_path = params["files"] + "/train/" + params["csv_file"]
    val_path = params["files"] + "/val/" + params["csv_file"]

    # Load the training and validation datasets
    train_files = glob.glob(train_path)
    val_files = glob.glob(val_path)

    # Load the bert tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(params["path_files"], do_lower_case=False)

    df_train = data_collector(train_files, params, True)
    df_val = data_collector(val_files, params, False)

    # Get the comment texts and corresponding labels
    if params["csv_file"] == "*_full.csv":
        sentences_train = df_train.text.values
        sentences_val = df_val.text.values
    elif params["csv_file"] == "*_translated.csv":
        sentences_train = df_train.translated.values
        sentences_val = df_val.translated.values

    labels_train = df_train.label.values
    labels_val = df_val.label.values
    label_counts = df_train["label"].value_counts()
    label_weights = [(len(df_train)) / label_counts[0], len(df_train) / label_counts[1]]

    # Select the required bert model. Refer below for explanation of the parameter values.
    model = select_model(params["what_bert"], params["path_files"], params["weights"])

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Do the required encoding using the bert tokenizer
    input_train_ids, att_masks_train = combine_preprocessing(
        sentences_train, tokenizer, params["max_length"]
    )
    # input_val_ids,att_masks_val=combine_preprocessing(sentences_val,tokenizer,params['max_length'])

    # Create dataloaders for both the train and validation datasets.
    train_dataloader = generate_dataloader(
        input_train_ids,
        labels_train,
        att_masks_train,
        batch_size=params["batch_size"],
        is_train=params["is_train"],
    )
    # validation_dataloader=generate_dataloader(input_val_ids,labels_val,att_masks_val,batch_size=params['batch_size'],is_train=False)

    # Initialize AdamW optimizer.
    optimizer = AdamW(
        model.parameters(),
        lr=params[
            "learning_rate"
        ],  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=params["epsilon"],  # args.adam_epsilon  - default is 1e-8.
    )

    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * params["epochs"]

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps / 10),  # Default value in run_glue.py
        num_training_steps=total_steps,
    )

    # Set the seed value all over the place to make this reproducible.
    fix_the_random(seed_val=params["random_seed"])
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # Create a new experiment in neptune for this run.
    bert_model = params["path_files"]
    language = params["language"]
    name_one = bert_model + "_" + language
    if params["logging"] == "neptune":
        neptune.create_experiment(
            name_one,
            params=params,
            send_hardware_metrics=False,
            run_monitoring_thread=False,
        )
        neptune.append_tag(bert_model)
        neptune.append_tag(language)

    # The best val fscore obtained till now, for the purpose of hyper parameter finetuning.
    best_val_fscore = best_val_fscore

    # For each epoch...
    for epoch_i in range(0, params["epochs"]):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, params["epochs"]))
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()
            # Get the model outputs for this batch.
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        if params["logging"] == "neptune":
            neptune.log_metric("avg_train_loss", avg_train_loss)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        # Compute the metrics on the validation and test sets.
        val_fscore, val_accuracy, val_auc, val_pred, val_rec = eval_phase(
            params, "val", model
        )
        test_fscore, test_accuracy, test_auc, test_pred, test_rec = eval_phase(
            params, "test", model
        )

        # Report the final accuracy and fscore for this validation run.
        if params["logging"] == "neptune":
            neptune.log_metric("val_fscore", val_fscore)
            neptune.log_metric("val_acc", val_accuracy)
            neptune.log_metric("val_auc", val_auc)
            neptune.log_metric("val_pred", val_pred)
            neptune.log_metric("val_rec", val_rec)
            neptune.log_metric("test_fscore", test_fscore)
            neptune.log_metric("test_accuracy", test_accuracy)
            neptune.log_metric("test_auc", test_auc)
            neptune.log_metric("test_pred", test_pred)
            neptune.log_metric("test_rec", test_rec)

        # Save the model only if the validation fscore improves. After all epochs, the best model is the final saved one.
        if val_fscore > best_val_fscore:
            print(val_fscore, best_val_fscore)
            best_val_fscore = val_fscore
            save_model(model, tokenizer, params)

    if params["logging"] == "neptune":
        neptune.stop()
    del model
    torch.cuda.empty_cache()
    return val_fscore, best_val_fscore


def create_divs(filename):
    df = pd.read_csv("Dataset/full_data/" + filename)
    if not os.path.exists("Dataset/train"):
        os.makedirs("Dataset/train")
    if not os.path.exists("Dataset/val"):
        os.makedirs("Dataset/val")
    if not os.path.exists("Dataset/test"):
        os.makedirs("Dataset/test")
    train_df_ids = list(pd.read_csv("Dataset/ID Mapping/train/" + filename)["id"])
    train_df = df.iloc[train_df_ids]
    train_df.to_csv("Dataset/train/" + filename, index=False)
    val_df_ids = list(pd.read_csv("Dataset/ID Mapping/val/" + filename)["id"])
    val_df = df.iloc[val_df_ids]
    val_df.to_csv("Dataset/val/" + filename, index=False)
    test_df_ids = list(pd.read_csv("Dataset/ID Mapping/test/" + filename)["id"])
    test_df = df.iloc[test_df_ids]
    test_df.to_csv("Dataset/test/" + filename, index=False)
