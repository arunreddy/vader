import scipy.io as sio
import pandas as pd

def load_meta_config(meta_path, filter_dataset='cat'):
  print(meta_path)
  data = sio.loadmat(meta_path)['nameMapping']
  rows = []
  for d in data:
    rows.append([x[0] for x in d.tolist()])
  df = pd.DataFrame(rows, columns=['img_name', 'common_name', 'dataset', 'train_test', 'class', 'is_flipped'])
  df['flipped'] = df.is_flipped.apply(lambda x: 1 if x == 'flipped' else 0)
  del df['is_flipped']

  df_dataset = df[df.dataset == filter_dataset]
  del df_dataset['dataset']
  return df_dataset
