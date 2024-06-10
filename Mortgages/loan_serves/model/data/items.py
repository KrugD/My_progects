import pandas as pd
df = pd.read_csv('30.6 homework.csv').drop('price_category', axis=1)


types = {
    'int64': 'int',
    'float64': 'float'
}
for k, v in df.dtypes.items():
    print(f'{k}: {types.get(str(v), "str")}')