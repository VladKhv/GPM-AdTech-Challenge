import joblib
from pathlib import Path
from shutil import copy

DATASET_PATH = Path(r'G:\python-datasets\user_conversions')
if not DATASET_PATH.is_dir():
    DATASET_PATH = Path(r'D:\python-datasets\user_conversions')

if not DATASET_PATH.exists():
    DATASET_PATH = Path('.')

MODELS_PATH = DATASET_PATH.joinpath('models')

# selected_models = ((93, [3]), (111, [0]), (117, [1, 3]))
# selected_models = ((205, [0, 1, 2, 3]), (117, [0, 1, 2, 3]))
# selected_models = ((195, [0, 2]), (303, [1, 2, 3]), (304, [1, 2, 3]))
selected_models = ((195, [0, 2, 3]),)
used_models = []
num_models = 0
for max_num, selected in selected_models:
    data_cls = joblib.load(MODELS_PATH.joinpath(f'data_cls_{max_num}.pkl'))
    models = joblib.load(MODELS_PATH.joinpath(f'models_{max_num}.pkl'))
    print(f'max_num: {max_num}, selected: {selected}, len(models) = {len(models)}')
    used = [models[i] for i in selected]
    used_models.append([data_cls, used])
    num_models += len(selected)

# сохраняем это в файлы
models_str = '_'.join([f"{num}_{'_'.join(map(str, idx))}" for num, idx in selected_models])

data_cls_pkl = MODELS_PATH.joinpath(f'data_cls_{models_str}.pkl')
models_pkl = MODELS_PATH.joinpath(f'models_{models_str}.pkl')

joblib.dump(num_models, data_cls_pkl, compress=7)
joblib.dump(used_models, models_pkl, compress=7)

copy(data_cls_pkl, r'G:\python-txt\user_conversions\models\data_cls.pkl')
copy(models_pkl, r'G:\python-txt\user_conversions\models\models.pkl')

print(models_str)
