import pandas as pd
import re
import argparse
import ast
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from catboost import CatBoostRegressor


def infer(file_path, output_file):
    # bad_lines = []

    chunksize = 10000
    chunks = pd.read_csv(file_path, chunksize=chunksize, encoding='utf-8', on_bad_lines='skip',
                         encoding_errors='replace')

    df_list = []
    for chunk in chunks:
        df_list.append(chunk)

    # Объединяем все прочитанные куски в один DataFrame
    df = pd.concat(df_list, ignore_index=True)

    print(f"Dataframe shape: {df.shape}")

    # Если у вас есть список грязных строк, вы можете сохранить их для анализа
    # with open("problematic_lines.txt", "w", encoding="utf-8") as f:
    #     for line in bad_lines:
    #         f.write(line)

    # Если нужно пропустить строки, не удалось прочитать
    chunks = pd.read_csv(file_path, chunksize=chunksize, encoding='utf-8', on_bad_lines='skip',
                         encoding_errors='ignore')

    df_list = []
    for chunk in chunks:
        df_list.append(chunk)

    df_clean = pd.concat(df_list, ignore_index=True)
    df.fillna("", inplace=True)  # меняем пустые значения на ""
    # Инициализация tqdm для pandas
    tqdm.pandas()

    # Столбцы по категориям
    useless = ['id', 'change_time', 'code_external_system', 'company_code', 'contact_person', 'data_ids', 'date_create',
              'date_modify', 'deleted', 'publication_period', 'published_date', 'salary_min', 'salary_max', 'salary',
              'state_region_code', 'contactList']
    categories = ['academic_degree', 'bonus_type', 'accommodation_type', 'measure_type', 'busy_type',
                  'career_perspective', 'code_professional_sphere', 'contact_source', 'education',
                  'foreign_workers_capability', 'is_mobility_program', 'is_moderated', 'is_uzbekistan_recruitment',
                  'is_quoted', 'need_medcard', 'original_source_type', 'company_business_size', 'retraining_capability',
                  'retraining_grant', 'schedule_type', 'source_type', 'status', 'transport_compensation']
    nums = ['additional_premium', 'code_profession', 'okso_code', 'required_experience', 'retraining_grant_value',
            'work_places', 'professionalSphereName']
    texts = ['additional_requirements', 'education_speciality', 'metro_ids', 'oknpo_code', 'other_vacancy_benefit',
             'position_requirements', 'position_responsibilities', 'regionName', 'regionNameTerm',
             'required_certificates', 'retraining_condition', 'social_protected_ids', 'vacancy_name',
             'federalDistrictCode', 'industryBranchName', 'company_name', 'full_company_name']
    keep_as_is = ['required_drive_license']
    label = pd.to_numeric(df.salary, errors='coerce')
    # Удаление ненужных столбцов (категория "useless")
    df = df.drop(columns=useless, errors='ignore')

    # Преобразование столбцов категорий nums в float и замена пустых значений на NaN
    for col in tqdm(nums):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Функция для удаления ключа owner_id из словаря
    def clean_dict(dict_str):
        try:
            dictionary = ast.literal_eval(dict_str)
            if isinstance(dictionary, dict) and 'owner_id' in dictionary:
                del dictionary['owner_id']
            return dictionary
        except (ValueError, SyntaxError):
            return {}

    # Добавление содержимого словарей в text_info
    skills_columns = ['languageKnowledge', 'hardSkills', 'softSkills']
    for col in skills_columns:
        df[col] = df[col].apply(lambda x: clean_dict(x))

    # Создание нового столбца text_info из текстовых столбцов и словарей
    text_info = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Building text_info column"):
        row_text = []
        for col in texts + skills_columns:
            if col in df.columns:
                value = row[col]
                if isinstance(value, dict) and value:  # Обработка словарей
                    row_text.append(f"{col}: {value}")
                else:  # Обработка остальных столбцов
                    row_text.append(f"{col}: {value}")
        text_info.append("\n ".join(row_text))

    df['text_info'] = text_info

    # Удаление текстовых столбцов и столбцов с данными словарей
    df = df.drop(columns=texts + skills_columns, errors='ignore')

    # Добавление новых столбцов A, B, C, D, E на основе required_drive_license
    def parse_license(x):
        try:
            license_list = ast.literal_eval(x)
            if isinstance(license_list, list):
                return license_list
            else:
                return []
        except (ValueError, SyntaxError):
            return []

    license_columns = ['A', 'B', 'C', 'D', 'E']
    for col in license_columns:
        df[col] = df['required_drive_license'].apply(lambda x: 1 if col in parse_license(x) else 0)

    # df = df[nums + categories + ["text_info"] +license_columns]
    df = df[nums + categories + ["text_info"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Инициализация токенизатора и модели ruBERT
    save_directory = './saved_RUSSIAN_bert_model'
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

    data = df
    # Применяем функцию к столбцу text_info
    tqdm.pandas()
    data['bert_embeddings'] = data['text_info'].progress_apply(get_bert_embeddings)
    data['bert_embeddings'] = data['bert_embeddings'].apply(lambda x: x.squeeze())  # Убираем лишние измерения
    # Преобразуем эмбеддинги в DataFrame
    embeddings_df = pd.DataFrame(data['bert_embeddings'].tolist(), index=data.index)

    # Объединяем с основными данными
    data = pd.concat([data, embeddings_df], axis=1)

    data.columns = [str(col) if isinstance(col, (int, float)) else col for col in data.columns]
    data = data.drop(columns=['text_info', 'bert_embeddings'])


    model = CatBoostRegressor()

    # Загрузка модели
    model.load_model('catboost_model_salary.cbm')

    # Теперь вы можете использовать модель для предсказаний
    predictions = model.predict(data)

    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])

    # Сохранение предсказаний в CSV-файл
    predictions_df.to_csv(output_file, index=False)

def main(input_path, submission_path):
    infer(input_path, submission_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer model for salary prediction.')
    parser.add_argument('input', type=str, help='Path to the input CSV file')
    parser.add_argument('submission', type=str, help='Path to save the submission or model file')

    args = parser.parse_args()

    main(args.input, args.submission)