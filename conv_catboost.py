import re
import os
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import joblib
from shutil import copy
from pathlib import Path
from datetime import datetime

import torch
import optuna
from optuna.integration import CatBoostPruningCallback
from catboost import CatBoostClassifier, Pool

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from df_addons import df_to_excel
from print_time import print_time, print_msg

from data_process_conv import DataTransform

__import__('warnings').filterwarnings("ignore")

DATASET_PATH = Path(r'G:\python-datasets\user_conversions')
if not DATASET_PATH.is_dir():
    DATASET_PATH = Path(r'D:\python-datasets\user_conversions')

if not DATASET_PATH.exists():
    DATASET_PATH = Path('.')
    __file__ = Path('.')
    LOCAL_FILE = ''
else:
    LOCAL_FILE = '_local'


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial: optuna.Trial) -> float:
    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "auto_class_weights": trial.suggest_categorical("auto_class_weights",
                                                        [None, "Balanced"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        # "iterations": trial.suggest_int("iterations", 200, 2000, step=200),
        # "depth": trial.suggest_int("depth", 1, 12),
        # "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 1.0),
        # "random_strength": trial.suggest_int("random_strength", 1, 20),
    }

    clf = CatBoostClassifier(
        cat_features=cat_features,
        eval_metric='F1',
        early_stopping_rounds=100,
        random_seed=SEED,
        task_type="GPU",
        **param
    )

    pruning_callback = CatBoostPruningCallback(trial, "F1")

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    accuracy = accuracy_score(y_valid, clf.predict(X_valid))
    return accuracy


def get_max_num(file_logs=None):
    """Получение максимального номера итерации обучения моделей
    :param file_logs: имя лог-файла с полным путем
    :return: максимальный номер
    """
    if file_logs is None:
        file_logs = DATASET_PATH.joinpath(f'scores{LOCAL_FILE}.logs')

    if not file_logs.is_file():
        with open(file_logs, mode='a') as log:
            log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
                      'model_columns;exclude_columns;cat_columns;comment\n')
        max_num = 0
    else:
        df = pd.read_csv(file_logs, sep=';')
        if 'acc_train' not in df.columns:
            df.insert(2, 'acc_train', 0)
            df.insert(3, 'acc_valid', 0)
            df.insert(4, 'acc_full', 0)
            df.num = df.index + 1
            df.to_csv(file_logs, index=False, sep=';')
        if 'roc_auc' not in df.columns:
            df.insert(2, 'roc_auc', 0)
            df.num = df.index + 1
            df.to_csv(file_logs, index=False, sep=';')
        df.num = df.index + 1
        max_num = df.num.max() if len(df) else 0
    return max_num


def predict_train_valid(model, datasets, submit_prefix='', label_enc=None):
    """Расчет метрик для модели: accuracy на трейне, на валидации, на всем трейне, roc_auc
    и взвешенная F1-мера на валидации
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param submit_prefix: префикс для файла сабмита для каждой модели свой
    :param label_enc: используемый label_encоder для target'а
    :return: accuracy на трейне, на валидации, на всем трейне, roc_auc и взвешенная F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target = datasets

    if 'label' in X_train.columns:
        X_train = X_train.copy().drop(['time', 'label'], axis=1)
    if 'label' in X_valid.columns:
        X_valid = X_valid.copy().drop(['time', 'label'], axis=1)
    if 'label' in train.columns:
        train = train.copy().drop(['time', 'label'], axis=1)

    valid_pred = model.predict(X_valid)
    train_pred = model.predict(X_train)
    train_full = model.predict(train)
    try:
        predict_proba = model.predict_proba(X_valid)[:, 1]
    except:
        predict_proba = train_pred

    ci_score = model.score(X_valid, y_valid)

    try:
        f1w = f1_score(y_valid, valid_pred)
        acc_valid = accuracy_score(y_valid, valid_pred)
        acc_train = accuracy_score(y_train, train_pred)
        acc_full = accuracy_score(target, train_full)
    except:
        f1w = acc_train = acc_full = 0
        acc_valid = ci_score
    try:
        roc_auc = roc_auc_score(y_valid, predict_proba)
    except:
        roc_auc = 0

    print(f"Score модели : {ci_score}")
    print(f'Weighted F1-score = {f1w:.6f}')
    print(f'Accuracy train:{acc_train} valid:{acc_valid} full:{acc_full} roc_auc:{roc_auc}')

    # print(classification_report(y_valid, valid_pred))
    return acc_train, acc_valid, acc_full, roc_auc, f1w, ci_score


def make_groups(row):
    if row.user_id in grp1:
        x = 1
    elif row.user_id in grp2:
        x = 2
    elif row.user_id in grp3:
        x = 3
    else:
        if row.record_count < 30:
            x = row.record_count + 4
        elif row.record_count < 35:
            x = 35
        elif row.record_count < 41:
            x = 41
        elif row.record_count < 51:
            x = 51
        else:
            x = 55
    return x


file_logs = DATASET_PATH.joinpath(f'scores{LOCAL_FILE}.logs')
max_num = get_max_num(file_logs)

VIEWS_PATH = DATASET_PATH.joinpath('train_views.parquet')
ACTIONS_PATH = DATASET_PATH.joinpath('train_actions.parquet')
THIRD_PARTY_PATH = DATASET_PATH.joinpath('third_party_conversions.parquet')

MODELS_PATH = DATASET_PATH.joinpath('models')

# проверка наличия GPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f'Обучаюсь на: {device}')

start_time = print_msg('Обработка third_party_conversions...')

conv_info_pkl = DATASET_PATH.joinpath('conv_info.pkl')
if conv_info_pkl.is_file():
    conv_info = pd.read_pickle(conv_info_pkl)
else:
    third_party = pd.read_parquet(THIRD_PARTY_PATH)
    third_party['time'] = pd.to_datetime(third_party['time'])
    third_party['ud_cookie_ts'] = pd.to_datetime(third_party['ud_cookie_ts'])

    conv_info = third_party.pivot_table(index='user_id',
                                        values='ua_os',
                                        columns='conversion_name',
                                        aggfunc='count', fill_value=0)
    conv_info.columns = [f'cn_{b}'.replace('implicit_depth_', 'dpt_')
                         for b in conv_info.columns]
    conv_info['total_count'] = conv_info.sum(axis=1)
    conv_info.reset_index(inplace=True)
    conv_info.to_pickle(conv_info_pkl)
    del third_party

print_time(start_time)
#

DEBUG = False

for max_features in [128]:

    start_time = print_msg('Обучение Catboost классификатор...')

    # Чтение трейна в ДФ
    train_views = pd.read_parquet(VIEWS_PATH)

    if DEBUG:
        train_views = train_views.iloc[:50000]

    print(f'Размер train_views = {train_views.shape}')

    train_actions = pd.read_parquet(ACTIONS_PATH)

    # third_party_conv = pd.read_parquet(THIRD_PARTY_PATH)

    flt = (train_actions.conversion_name == 'cart') & (train_actions.is_post_click == 1)
    train_df = train_views.merge(train_actions[flt][['ssp_event_id', 'is_post_click']],
                                 how='left', on='ssp_event_id')

    train_df.is_post_click.fillna(0, inplace=True)
    train_df.drop('ssp_event_id', inplace=True, axis=1)

    # rename target column
    train_df.rename(columns={'is_post_click': 'label'}, inplace=True)
    train_df.label = train_df.label.astype(int)

    # удалить дубликаты после объединения
    drop_duplicates = True

    if drop_duplicates:
        train_df.drop_duplicates(inplace=True)

    print(f'Размер train_df = {train_df.shape}', 'пропусков:', train_df.isna().sum().sum())

    original_columns = train_df.columns.to_list()

    data_cls = DataTransform(use_catboost=True, drop_first=False)

    data_cls.exclude_columns = [
        'ssp_event_id',
        'bid_ip',  # #
        # 'bid_referer',
        # 'page_language',  # можно оставить это
        # 'battr',
        # 'model',  # можно оставить это
        # 'mobile_screen_size',
        # 'mime_types',
        # 'historical_viewability',  # можно оставить это
        # 'do_not_track',
        # 'is_mobile_optimized_site',
        # 'device_screen',
        # 'utm_source',
        # 'search_engine',
        # 'search_terms',
        # 'ud_cookie_ts',
        # 'accept_encoding',
        # 'accept_language',
        # 'full_placement_id2',  # это поле почти полностью повторяет ssp
        'user_segments',  # #
        'record_catalog',
        'known',
    ]

    # Результат 70 модели без векторизации roc_auc: 0,938773
    # Результат 64 модели  с векторизацией roc_auc: 0,943513
    # задать тип векторизации
    # data_cls.vector_limit = 0
    # data_cls.vector_limit = 32
    # data_cls.vectorizer = CountVectorizer
    # data_cls.vectorizer = TfidfVectorizer
    # data_cls.ngram_range = (1, 2)
    data_cls.ngram_range = (1, 1)
    data_cls.min_df = 1
    data_cls.max_features = 128
    # data_cls.max_features = max_features

    # удалять из трейна юзеров с пропусками
    # data_cls.drop_nan_users = True  # Это ухудшает скор на -0.1
    # удалять из трейна юзеров с пропусками только с X записями
    # data_cls.drop_nan_users_with_records = 2  # ставим количество записей

    # Максимальный предел пропусков для нахождения моды и среднего по умолчанию = 0.2 (20%)
    # data_cls.max_limit_nans = 1.1  # т.е. все считаем по моде и среднему

    # добавление информации по third_party_conversions
    # при добавлении этой инфы скор локальный и на платформе падает
    # data_cls.conv_info = conv_info

    train_df = data_cls.fit_transform(train_df.copy())

    cat_features = data_cls.category_columns
    num_features = data_cls.numeric_columns

    print(f'Размер train_df = {train_df.shape}', 'пропусков:', train_df.isna().sum().sum())

    # test_sizes = (0.3,)
    test_sizes = (0.4,)
    # test_sizes = (0.38, 0.39)
    # test_sizes = np.linspace(0.3, 0.4, 3)
    # test_sizes = np.linspace(0.25, 0.35, 3)
    # for num_iters in range(500, 701, 50):
    # for SEED in range(100):
    for test_size in test_sizes:
        # for num_folds in (3, 4, 6, 7):
        # for depth in range(1, 6):
        # for num_leaves in range(20, 51, 5):
        max_num += 1

        # test_size = 0.37

        SEED = 17

        num_iters = 7000

        num_folds = 4

        test_size = round(test_size, 2)

        split_date = datetime(2024, 1, 19)

        split_to_users_grp = True
        fit_on_full_train = False
        use_grid_search = False
        use_split_date = False
        use_cv_folds = True
        build_model = True
        stratified = None

        np.random.seed(SEED)

        # Разделение на обучающую и валидационную выборки

        train = train_df.drop(['time', 'label'], axis=1)
        target = train_df.label

        model_columns = train.columns.to_list()
        exclude_columns = [col for col in original_columns if col not in model_columns]

        data_cls.model_columns = model_columns
        data_cls.exclude_columns = exclude_columns

        print('Обучаюсь на колонках:', model_columns)
        print('Категорийные колонки:', cat_features)
        print('Исключенные колонки:', exclude_columns)

        # сохранение класса data_cls в файл с использованием компрессии уровня 7
        data_cls_pkl = MODELS_PATH.joinpath(f'data_cls_{max_num}.pkl')
        joblib.dump(data_cls, data_cls_pkl, compress=7)
        copy(data_cls_pkl, r'G:\python-txt\user_conversions\models\data_cls.pkl')

        # if data_cls.vectorizer is not None:
        #     # Сохраняем обученный векторайзер в файл с использованием компрессии уровня 7
        #     joblib.dump(data_cls.bigram_vectorizer,
        #                 MODELS_PATH.joinpath(f'vectorizer_{max_num}.joblib'), compress=7)

        # exit()

        if use_split_date:
            # (Timestamp('2024-01-02 21:00:03'), Timestamp('2024-01-23 23:59:57'))
            # вариант разбиения по дате
            X_train = train_df[(train_df.time < split_date)].drop(['time', 'label'], axis=1)
            y_train = train_df.label[(train_df.time < split_date)]
            X_valid = train_df[(train_df.time > split_date)].drop(['time', 'label'], axis=1)
            y_valid = train_df.label[(train_df.time > split_date)]

        else:
            if split_to_users_grp:
                stratified = ['label', 'rec_grp']

                start_grp = print_msg('Группировка данных...')

                users = train_df[['user_id', 'label', 'record_count']].drop_duplicates()

                users_labels = users[users.label == 1]

                grp1 = users_labels[users_labels.record_count < 10].user_id
                grp2 = users_labels[(users_labels.record_count > 9) &
                                    (users_labels.record_count < 31)].user_id
                grp3 = users_labels[users_labels.record_count > 30].user_id
                users['rec_grp'] = users.apply(lambda row: make_groups(row), axis=1)

                users['grp_label'] = users.apply(
                    lambda row: '_'.join(str(row[col]) for col in stratified), axis=1)
                users.loc[users.user_id.isin(grp1), 'grp_label'] = '1_1'
                users.loc[users.user_id.isin(grp2), 'grp_label'] = '1_2'
                users.loc[users.user_id.isin(grp3), 'grp_label'] = '1_3'

                # экспорт в эксель
                # df_to_excel(users, DATASET_PATH.joinpath('users.xlsx'))

                # train_df.drop('record_count', axis=1, inplace=True)

                print('users.grp_label.value_counts()')
                print(users.grp_label.value_counts())

                # вариант разбиения по юзерам

                U_train, U_valid = train_test_split(users, test_size=test_size,
                                                    random_state=SEED,
                                                    stratify=users.grp_label)

                X_train = train_df[train_df.user_id.isin(U_train.user_id)]
                X_valid = train_df[train_df.user_id.isin(U_valid.user_id)]

                print_time(start_grp)

            else:
                stratified = ['label']

                X_train, X_valid = train_test_split(train_df, test_size=test_size,
                                                    random_state=SEED,
                                                    stratify=train_df.label)

            y_train = X_train.label
            y_valid = X_valid.label

            X_train.drop(['time', 'label'], axis=1, inplace=True)
            X_valid.drop(['time', 'label'], axis=1, inplace=True)
            #

            real_size = round(X_valid.shape[0] / train_df.shape[0], 2)
            test_size = f'{test_size} real_size: {real_size}'

        if use_cv_folds:
            test_size = f'folds={num_folds}'
        else:
            print('X_train.shape', X_train.shape, 'пропусков:', X_train.isna().sum().sum())
            print('X_valid.shape', X_valid.shape, 'пропусков:', X_valid.isna().sum().sum())

        print(f'test_size: {test_size} SEED={SEED} split_date: {split_date}')

        pool_train = Pool(data=X_train, label=y_train, cat_features=cat_features)
        pool_valid = Pool(data=X_valid, label=y_valid, cat_features=cat_features)

        skf = StratifiedKFold(n_splits=num_folds, random_state=SEED, shuffle=True)
        split_kf = KFold(n_splits=num_folds, random_state=SEED, shuffle=True)

        models, models_scores, predict_scores = [], [], []

        splited = X_train, X_valid, y_train, y_valid

        # auto_class_weights = 'Balanced'
        auto_class_weights = None

        # минимальное количество итераций при обучении модели
        min_iterations = 300 - 230 * (auto_class_weights is not None)

        loss_function = 'Logloss'

        eval_metric = 'AUC:hints=skip_train~false'
        # eval_metric = 'TotalF1'
        # eval_metric = 'Accuracy'

        clf_params = dict(cat_features=cat_features,
                          # scale_pos_weight=4,
                          auto_class_weights=auto_class_weights,
                          loss_function=loss_function,
                          eval_metric=eval_metric,
                          # depth=4,
                          # learning_rate=0.02,
                          iterations=num_iters,  # попробовать столько итераций
                          early_stopping_rounds=100,
                          random_seed=SEED,
                          task_type="GPU" if device == "cuda" else "CPU",
                          # devices='0:1',
                          )

        clf = CatBoostClassifier(**clf_params)

        kind_fit = 'train'

        if use_grid_search:
            # grid_params = {
            #     'max_depth': [5, 6],
            #     'learning_rate': [0.1, 0.15, 0.2],
            # }
            # grid_search_result = clf.grid_search(grid_params, train, target,
            #                                      cv=skf,
            #                                      stratified=True,
            #                                      refit=True,
            #                                      plot=True,
            #                                      verbose=100,
            #                                      )
            # best_params = grid_search_result['params']
            # models.append(clf)

            # Выполнить оптимизацию гиперпараметров
            study = optuna.create_study(
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                direction="maximize"
            )
            study.optimize(objective, n_trials=40, timeout=600)

            print("Количество завершенных испытаний: {}".format(len(study.trials)))
            print("Лучшее испытание:")
            trial = study.best_trial
            print("  Значение: {}".format(trial.value))
            print("  Параметры: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            best_params = trial.params

            clf_params.update(best_params)
            print('clf_params', clf_params)

            clf = CatBoostClassifier(**clf_params)

        if use_cv_folds:
            if split_to_users_grp:
                if stratified:
                    if len(stratified) > 1 and isinstance(stratified, (list, tuple)):
                        stratified_cols = users.grp_label
                    else:
                        stratified_cols = users.label

                    skf_folds = skf.split(users, stratified_cols)

                else:
                    skf_folds = split_kf.split(users)
            else:
                if stratified:
                    skf_folds = skf.split(train_df, train_df.label)
                else:
                    skf_folds = split_kf.split(train_df)

            for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
                print(f'Фолд {idx} из {num_folds}')
                if split_to_users_grp:
                    X_train = train_df[train_df.user_id.isin(users.iloc[train_idx].user_id)]
                    X_valid = train_df[train_df.user_id.isin(users.iloc[valid_idx].user_id)]
                else:
                    X_train = train_df.iloc[train_idx]
                    X_valid = train_df.iloc[valid_idx]

                y_train = X_train.label
                y_valid = X_valid.label

                X_train.drop(['time', 'label'], axis=1, inplace=True)
                X_valid.drop(['time', 'label'], axis=1, inplace=True)

                splited = X_train, X_valid, y_train, y_valid

                train_data = Pool(data=X_train, label=y_train, cat_features=cat_features)
                valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_features)

                clf = CatBoostClassifier(**clf_params)
                clf.fit(train_data, eval_set=valid_data, use_best_model=True, verbose=100)

                # не будем включать в список модели с малым количество итераций
                if DEBUG or (not DEBUG and clf.best_iteration_ > min_iterations):
                    models.append(clf)

                if build_model:
                    DTS = (*splited, train, target)
                    predict_scores = predict_train_valid(clf, DTS)

                    # не будем включать в список модели с малым количество итераций
                    if DEBUG or (not DEBUG and clf.best_iteration_ > min_iterations):
                        models_scores.append(predict_scores)

                    acc_train, acc_valid, acc_full, roc_auc, f1w, score = predict_scores

                    real_size = round(len(X_valid) / (len(X_valid) + len(X_train)), 2)
                    fold_size = f'{test_size} real_size: {real_size}'
                    comment = {
                        'test_size': str(split_date.date()) if use_split_date else fold_size,
                        'split_grp': split_to_users_grp,
                        'drop_dupl': drop_duplicates,
                        'SEED': SEED,
                        'kind_fit': f'pool_{idx}',
                        'clf_iters': clf.best_iteration_,
                        'clf_lr': clf.get_params().get('learning_rate'),
                    }
                    if DEBUG:
                        comment['DEBUG'] = DEBUG
                    comment.update(data_cls.comment)
                    comment.update({'stratified': stratified})
                    comment.update(clf.get_params())

                    with open(file_logs, mode='a') as log:
                        # log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
                        #           'model_columns;exclude_columns;cat_columns;comment\n')
                        log.write(
                            f'{max_num};cb;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                            f'{acc_full:.6f};'
                            f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                            f'{exclude_columns};{cat_features};{comment}\n')

            best_params = {'iterations': [clf.best_iteration_ for clf in models]}

        else:
            DTS = (*splited, train, target)

            clf.fit(pool_train, eval_set=pool_valid, use_best_model=True, verbose=100)

            models.append(clf)

            best_params = {'clf_iters': clf.best_iteration_,
                           'clf_lr': clf.get_all_params()['learning_rate']}

            if build_model:
                if not fit_on_full_train:
                    predict_scores = predict_train_valid(clf, DTS)

                else:
                    predict_scores = predict_train_valid(clf, DTS)
                    acc_train, acc_valid, acc_full, roc_auc, f1w, score = predict_scores
                    comment = {
                        'test_size': str(split_date.date()) if use_split_date else test_size,
                        'split_grp': split_to_users_grp,
                        'drop_dupl': drop_duplicates,
                        'SEED': SEED,
                        'kind_fit': kind_fit,
                        'clf_iters': clf.best_iteration_,
                        'clf_lr': clf.get_params().get('learning_rate'),
                    }
                    if DEBUG:
                        comment['DEBUG'] = DEBUG
                    comment.update(data_cls.comment)
                    comment.update({'stratified': stratified})
                    comment.update(models[0].get_params())

                    with open(file_logs, mode='a') as log:
                        # log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
                        #           'model_columns;exclude_columns;cat_columns;comment\n')
                        log.write(
                            f'{max_num};cb;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                            f'{acc_full:.6f};'
                            f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                            f'{exclude_columns};{cat_features};{comment}\n')

                    print('Обучаюсь на всём трейне...')
                    clf_params['iterations'] = int(clf.tree_count_ * 1.1)
                    clf_params['learning_rate'] = clf.get_all_params()['learning_rate']
                    model = CatBoostClassifier(**clf_params)
                    model.fit(train, target)
                    predict_scores = predict_train_valid(model, DTS)
                    kind_fit = 'full'

            best_params.update(clf.get_params())

        print('best_params:', best_params)

        if build_model:
            if len(models) > 1:
                predict_scores = [np.mean(arg) for arg in zip(*models_scores)]

            acc_train, acc_valid, acc_full, roc_auc, f1w, score = predict_scores

            print(f'Weighted F1-score = {f1w:.6f}')
            print('Параметры модели:', clf.get_params())

            print_time(start_time)

            if len(models):
                clf_iters = [clf.best_iteration_ for clf in models]
                clf_lr = [clf.get_params().get('learning_rate') for clf in models]
                if len(models) == 1:
                    clf_iters = clf_iters[0]
                    clf_lr = clf_lr[0]

                # построим важность признаков
                feature_importance = 0
                for model in models:
                    feature_importance += np.array(model.get_feature_importance())
                feature_importance = feature_importance / len(models)

                feature_names = np.array(model_columns)
                data = {'feature_names': feature_names,
                        'feature_importance': feature_importance}
                fi_df = pd.DataFrame(data)
                fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
                fi_df.reset_index(drop=True, inplace=True)

                fi_df['type'] = ''
                fi_df.loc[fi_df['feature_names'].isin(cat_features), 'type'] = 'cat'
                fi_df.loc[fi_df['feature_names'].isin(num_features), 'type'] = 'num'
                fi_df.to_csv(MODELS_PATH.joinpath(f'feature_names_{max_num}.csv'), sep=';',
                             index=False)
            else:
                clf_iters = None
                clf_lr = None

            comment = {'test_size': str(split_date.date()) if use_split_date else test_size,
                       'split_grp': split_to_users_grp,
                       'drop_dupl': drop_duplicates,
                       'SEED': SEED,
                       'kind_fit': kind_fit,
                       'clf_iters': clf_iters,
                       'clf_lr': clf_lr,
                       }
            if DEBUG:
                comment['DEBUG'] = DEBUG
            comment.update({'stratified': stratified})
            comment.update(data_cls.comment)
            if models:
                comment.update(models[0].get_params())

            with open(file_logs, mode='a') as log:
                # log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
                #           'model_columns;exclude_columns;cat_columns;comment\n')
                log.write(f'{max_num};cb;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                          f'{acc_full:.6f};'
                          f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                          f'{exclude_columns};{cat_features};{comment}\n')

            start_time = print_msg('Сохранение моделей...')
            models_pkl = MODELS_PATH.joinpath(f'models_{max_num}.pkl')
            joblib.dump(models, models_pkl, compress=7)
            copy(models_pkl, r'G:\python-txt\user_conversions\models\models.pkl')
            print_time(start_time)
