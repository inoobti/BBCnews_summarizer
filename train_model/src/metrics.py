from transformers import EvalPrediction
import evaluate
import torch
from model import tokenizer

bleu_m = []
rouge1_m = []
rouge2_m = []
rougeL_m = []
rougeL_sum_m = []
meteor_m = []

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")


def filter_invalid_tokens(predictions):
    return [
        [token if 0 <= token < tokenizer.vocab_size else tokenizer.pad_token_id for token in seq]
        for seq in predictions
    ]

def compute_metrics(p: EvalPrediction):

    logits, labels = p
    decoded_preds = tokenizer.batch_decode(filter_invalid_tokens(logits), skip_special_tokens=True)

    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    labels = labels.detach().to(device)

    labels[labels == -100] = tokenizer.pad_token_id

    labels_list = labels.tolist()

    decoded_labels = tokenizer.batch_decode(labels_list, skip_special_tokens=True)

    decoded_labels = [[label] for label in decoded_labels]
    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

    bleu_m.append(round(bleu["bleu"], 4))
    rouge1_m.append(round(rouge['rouge1'], 4))
    rouge2_m.append(round(rouge['rouge2'], 4))
    rougeL_m.append(round(rouge['rougeL'], 4))
    rougeL_sum_m.append(round(rouge['rougeLsum'], 4))
    meteor_m.append(round(meteor['meteor'], 4))

    return {
        "bleu": bleu["bleu"],
        **rouge,
        **meteor,}