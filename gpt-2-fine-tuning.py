# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -

df = pd.read_csv('/kaggle/input/3k-conversations-dataset-for-chatbot/Conversation.csv')
df.head()

df=df.drop('Unnamed: 0',axis=1)
df.isnull().sum()

# +
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset

df['input'] = df.apply(lambda row: f"Question: {row['question']} Answer: {row['answer']}", axis=1)

# Convert the DataFrame to a Dataset object
dataset = Dataset.from_pandas(df[['input']])

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token if it does not exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['input'], padding='max_length', truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# +
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir='./logs',
)

# +
# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Update model embeddings to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# + _kg_hide-input=true
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()
# -

# Save the fine-tuned model
model.save_pretrained('fine_tuned_gpt2')
tokenizer.save_pretrained('fine_tuned_gpt2')

# +
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

# Load your fine-tuned model
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2')

# +
# Custom prompt
prompt = "Hi,what do you think we should do now?"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text based on the prompt
generated = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1, temperature=1.0)

# Decode the generated sequence of tokens back into text
generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

print("Generated text:\n", generated_text)

# +
# Custom prompt
prompt = "Do you think donald trump faked his assasination"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text based on the prompt
generated = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1, temperature=1.0)

# Decode the generated sequence of tokens back into text
generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

print("Generated text:\n", generated_text)
# -


