import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from catboost import CatBoostClassifier
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel


def infer(file_path):

    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)

    df['achievements'] = np.where(df['achievements'] == df['achievements_modified'], "", df['achievements'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка данных

    # Инициализация токенизатора и модели ruBERT
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased').to(device)  # Перемещение модели на GPU
    model.eval()  # Установка модели в режим оценки

    def get_bert_embeddings(text):
        # Разделение текста на подстроки, если он длиннее 512 символов
        max_length = 512
        embeddings = []

        # Разделяем текст на подстроки
        for i in range(0, len(text), max_length):
            sub_text = text[i:i + max_length]
            inputs = tokenizer(sub_text, return_tensors='pt', truncation=True, padding=True).to(
                device)  # Перемещение входных данных на GPU

            with torch.no_grad():
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0,
                                :].cpu().numpy()  # Получаем CLS токен и перемещаем обратно на CPU
                embeddings.append(cls_embedding)

        # Складываем эмбеддинги, если есть несколько подстрок
        if len(embeddings) > 1:
            return np.sum(embeddings, axis=0)
        else:
            return embeddings

    tqdm.pandas()
    for column in df.columns:
        df['bert_embeddings' + column] = df[column].progress_apply(get_bert_embeddings)

