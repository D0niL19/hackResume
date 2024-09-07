import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from catboost import CatBoostClassifier
from tqdm import tqdm
import torch
import argparse
from transformers import BertTokenizer, BertModel


def infer(file_path, submission_path):

    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)

    df['achievements'] = np.where(df['achievements'] == df['achievements_modified'], "", df['achievements'])

    # Проверка наличия CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка данных
    save_directory = './saved_RUSSIAN_bert_model'

    # Инициализация токенизатора и модели ruBERT
    tokenizer = BertTokenizer.from_pretrained(save_directory)
    model = BertModel.from_pretrained(save_directory).to(device)  # Перемещение модели на GPU
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
            return embeddings[0]

    # Применяем функцию к столбцам
    tqdm.pandas()
    for column in df.columns:
        if column in {'id_cv', 'job_title'}:
            continue
        df['bert_embeddings_' + column] = df[column].progress_apply(get_bert_embeddings)

    new_data = pd.DataFrame()

    new_data['bert_embeddings_achievements'] = df['bert_embeddings_achievements'].apply(
        lambda x: x.squeeze())
    new_data['bert_embeddings_achievements_modified'] = df[
        'bert_embeddings_achievements_modified'].apply(lambda x: x.squeeze())
    new_data['bert_embeddings_company_name'] = df['bert_embeddings_company_name'].apply(
        lambda x: x.squeeze())
    new_data['bert_embeddings_demands'] = df['bert_embeddings_demands'].apply(lambda x: x.squeeze())
    new_data['job_title'] = df['job_title']

    res_embedd = []
    for column in new_data.columns:
        if column in {'job_title', 'id_cv'}:
            continue
        embeddings_df = pd.DataFrame(new_data[column].tolist(), index=new_data.index)
        embeddings_df.columns = [column + str(col) for col in embeddings_df.columns]
        res_embedd.append(embeddings_df)

    data = pd.concat(res_embedd, axis=1)

    model = CatBoostClassifier()

    # Загрузка модели
    model.load_model('catboost_model_resume.cbm')

    predictions = model.predict(data)

    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])

    # Сохранение предсказаний в CSV-файл
    predictions_df.to_csv(submission_path, index=False)

def main(input_path, submission_path):
    infer(input_path, submission_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer model for job classification.')
    parser.add_argument('input', type=str, help='Path to the input CSV file')
    parser.add_argument('submission', type=str, help='Path to save the submission or model file')

    args = parser.parse_args()

    main(args.input, args.submission)