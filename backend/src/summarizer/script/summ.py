from transformers import AutoModelForSeq2SeqLM, MBart50TokenizerFast
from pathlib import Path

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SummarizerProcessor:
    def __init__(self, ):

        local_path = Path(__file__).resolve().parent.parent.parent.parent
        self.model_path = local_path / 'model' / 'bart'

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(device)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_path)

        self.text = str()

    def tokenize(self,):

        inputs = self.tokenizer(
            self.text,
            add_special_tokens=True,
            max_length=512,
            # pad_to_max_length=True,
            return_token_type_ids=True,
            return_tensors="pt",
            truncation=True,
        )

        return inputs

    def inference(self, input_text: str):

        self.text = input_text
        tokenized_inputs = self.tokenize()
        input_ids = tokenized_inputs["input_ids"]
        output = self.model.generate(
            input_ids=input_ids, max_length=128, num_beams=4, early_stopping=True,
        )
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return decoded_output

