"""
This project fine-tune a GPT-2 model to be a zombie screenwriter using Hugging Face’s transformers package.


Thanks to the team at @huggingface for publishing an example for fine_tuning a dataset,
and thanks to @cdpierse for creating the project "script_buddy_v2", this borrows heavily from those projects:
https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py
https://github.com/cdpierse/script_buddy_v2
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import AutoTokenizer, AutoModelWithLMHead
import numpy as np
import os
import random
from language_modelling import ScriptData
from utils import load_model, generate

output_dir = 'models/'

model_name = 'gpt2/'
model_dir = 'models/'
tuned_model_dir = 'models_tuned/'

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.00002
WARMUP_STEPS = 10000

# Function to first select topN tokens from the probability list and then based on the selected N word distribution
# get random token ID
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def load_model(model_dir, model_name):
    """Loads the saved GPT2 model from disk if the directory exists.
    Otherwise it will download the model and tokenizer from hugging face.
    Returns
    a tuple consisting of `(model,tokenizer)`
    """
    model_dir = os.path.join(model_dir, model_name)
    if not os.path.isdir(model_dir):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        os.makedirs(model_dir)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        print('model already there')
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelWithLMHead.from_pretrained(model_dir)
    return model, tokenizer


def generate_some_text(input_str, text_len = 250):

    cur_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0).long().to(device)

    model.eval()
    with torch.no_grad():

        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(only one) batch and the last predicted embedding
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=10) #Randomly(from the given probability distribution) choose the next word from the top n words
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word

        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        print(output_text)


def loader():
    return load_model()


if __name__ == '__main__':

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    model = model.to(device)

    # generate_some_text(" Artificial general intelligence is ")

# tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
# tokenizer = GPT2Tokenizer.from_pretrained(os.path.join("storage", "models"))
# model = GPT2LMHeadModel.from_pretrained(output_dir)
# model = model.to(device)

    FILE_PATH = os.path.join("text_data", "The_Crisis_on_Spieltruppestrasse.txt")
    dataset = ScriptData(tokenizer=tokenizer, file_path=FILE_PATH)
    script_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 0.00002
    WARMUP_STEPS = 10000

    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)
    script_count = 0
    sum_loss = 0.0
    batch_count = 0

    for epoch in range(1, EPOCHS):
        print(f"EPOCH {epoch} started" + '=' * 30)
        for idx, script in enumerate(script_loader):
            print(f'idx: {idx}')
            outputs = model(script.to(device), labels=script.to(device))

            loss, logits = outputs[:2]
            loss.backward()
            sum_loss = sum_loss + loss.detach().data

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
            batch_count += 1
            print(batch_count)

            # script_count = script_count + 1
            # if script_count == BATCH_SIZE:
            #     script_count = 0
            #     batch_count += 1
            #     optimizer.step()
            #     scheduler.step()
            #     optimizer.zero_grad()
            #     model.zero_grad()

            # if batch_count == 200:
        print(f"sum loss: {sum_loss}")

        if epoch % 20 == 0:
            model.eval()
            sample_outputs = model.generate(bos_token_id=random.randint(1, 30000),
                                            do_sample=True,
                                            top_k=50,
                                            max_length=1000,
                                            top_p=0.95,
                                            num_return_sequences=1
                                            )
            print("Output:\n" + 100 * '-')
            for i, sample_output in enumerate(sample_outputs):
                print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            print("Output:\n" + 100 * '-')
            model.train()
        sum_loss = 0.0


    save_model_dir = os.path.join(tuned_model_dir, model_name)
    if not os.path.isdir(save_model_dir):
        os.makedirs(save_model_dir)
    output_model_file = os.path.join(save_model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_model_dir, CONFIG_NAME)

    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_model_dir)

