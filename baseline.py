import numpy as np
import pandas as pd
import joblib

VIEWS_PATH = './data/private_info/train_views.parquet'
ACTIONS_PATH = './data/private_info/train_actions.parquet'
THIRD_PARTY_PATH = './data/private_info/third_party_conversions.parquet'


def run():
    return [0] * 13


def make_pedictions(test_df_path):
    used_models = joblib.load('./models/models.pkl')
    data_cls = joblib.load('./models/data_cls.pkl')

    test_df = pd.read_parquet(test_df_path)

    # если в data_cls передано количество моделей --> другой процесс обработки
    if isinstance(data_cls, int):
        num_models = data_cls
    else:
        num_models = len(used_models)
        used_models = [[data_cls, used_models[:]]]

    # создание массива для сбора предсказаний моделей
    predictions = np.zeros((num_models, len(test_df)))
    print(f"Models count: {num_models}")

    # заполнение массива предсказаниями моделей
    idx = 0
    for cls_dt, models in used_models:
        # обработка данных по алгоритму данного класса
        df = cls_dt.transform(test_df.copy())
        # предсказание для списка моделей
        for clf in models:
            predictions[idx] = clf.predict_proba(df[cls_dt.model_columns])[:, 1]
            idx += 1

    # вычисление статистик по строкам массива предсказаний
    # final_predictions = np.mean(predictions, axis=0)
    # final_predictions = np.median(predictions, axis=0)
    # final_predictions = np.max(predictions, axis=0)

    # операции по строкам массива predictions если в строке есть значение больше 0.5,
    # выбрать максимум строки, если все значения меньше 0.5 -> посчитать среднее по строке
    final_predictions = np.where(np.any(predictions >= 0.5, axis=0),
                                 np.max(predictions, axis=0),
                                 np.max(predictions, axis=0))

    # df['pred'] = final_predictions
    # df['grp_pred'] = df.groupby('user_id')['pred'].transform('max').fillna(0)
    # final_predictions = df['grp_pred'].to_numpy()

    print(f"Predictions length: {len(final_predictions)}")
    return final_predictions


if __name__ == "__main__":
    pass
