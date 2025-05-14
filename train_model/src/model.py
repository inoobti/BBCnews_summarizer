from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
import torch
from huggingface_hub import login
import os


access_token = os.getenv('ACCESS_TOKEN')
login(token=access_token)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_list = ['cointegrated/rut5-base-multitask',
              'facebook/mbart-large-50',
              'ai-forever/FRED-T5-large',
              'csebuetnlp/mT5_multilingual_XLSum',
              'google/mt5-base']
model_name=model_list[1]
model_n = 'mbart'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer)