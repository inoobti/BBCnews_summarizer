from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from model import model, model_n, tokenizer, data_collator
from news_dataset_loader import NewsLoader
from metrics import compute_metrics
import string
import nltk

nltk.download('punkt_tab')

batch_size = 8  # Количество примеров на одно устройство (GPU/CPU) во время тренировки и валидации
model_dir = f"../backend/model/{model_n}"  # Путь для сохранения модели
eval_steps = logging_steps = 100


class DataPreparetion:
    def __init__(self, max_input_length=512, max_target_length=128, prefix="суммаризация: ", crop=False):
        self.prefix = prefix
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.nl = NewsLoader()
        self.data = self.nl.load_data()
        if crop:
            self.crop_data()

    def crop_data(self):
        self.data['train'] = self.data['train'].select(range(10000))
        self.data['test'] = self.data['test'].select(range(1000))
        self.data['validation'] = self.data['validation'].select(range(1000))

    @staticmethod
    def clean_text(text):
        sentences = nltk.sent_tokenize(text.strip())
        sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
        sentences_cleaned_no_titles = [
            sent for sent in sentences_cleaned
            if len(sent) > 0 and sent[-1] in string.punctuation
        ]
        text_cleaned = "\n".join(sentences_cleaned_no_titles)
        return text_cleaned

    def tokeniz_data(self, examples, prefix='summorize: '):
        if "text" not in examples or "summary" not in examples:
            raise ValueError("Input examples must contain 'text' and 'title' fields.")

        texts_cleaned = [self.clean_text(text) for text in examples["text"]]
        inputs = [prefix + text for text in texts_cleaned]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=128, truncation=True)

        # Устанавливаем -100 для игнорируемых токенов в labels
        labels["input_ids"] = [id if id != tokenizer.pad_token_id else -100 for id in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def get_data(self):
        medium_datasets_cleaned = self.data.filter(
            lambda example: (len(example['text']) >= 100) and (len(example['summary']) >= 20))
        tokenized_datasets = medium_datasets_cleaned.map(self.tokeniz_data, batched=True)
        return tokenized_datasets


dp = DataPreparetion(crop=True)
data = dp.get_data()

args = Seq2SeqTrainingArguments(
    model_dir,  # Директория для сохранения модели и чекпоинтов
    eval_strategy="steps",  # Оценка модели будет выполняться через равные промежутки шагов (batch updates)
    eval_steps=eval_steps,  # Частота оценки (каждые 100 шагов)

    logging_strategy="steps",  # Логирование также происходит через равные промежутки шагов
    logging_steps=logging_steps,  # Частота логирования (каждые 100 шагов)

    save_strategy="steps",  # Сохранение чекпоинтов модели выполняется через шаги
    save_steps=200,  # Модель будет сохраняться каждые 200 шагов

    learning_rate=4e-5,  # Скорость обучения (оптимальный баланс между скоростью сходимости и стабильностью)

    per_device_train_batch_size=batch_size,  # Размер батча на одно устройство во время обучения
    per_device_eval_batch_size=batch_size,  # Размер батча на одно устройство во время валидации

    weight_decay=0.01,  # Коэффициент регуляризации (борьба с переобучением)

    save_total_limit=1,  # Хранить не более 3-х последних чекпоинтов, чтобы экономить место

    num_train_epochs=1,  # Количество эпох обучения (проходов по всему датасету)

    predict_with_generate=True,  # Использовать `generate()` для предсказаний (важно для задач генерации текста)

    bf16=True,  # Использовать вычисления в 32-битном формате (ускоряет обучение на современных GPU)

    load_best_model_at_end=True,  # Загружать лучшую модель по окончании обучения

    metric_for_best_model="eval_bleu",  # Метрика, по которой выбирается лучшая модель (важно для задач суммаризации)
)

trainer = Seq2SeqTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model(model_dir)
