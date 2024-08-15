import pandas as pd
import json
import dill
import os
import datetime

paths = os.environ.get('$PROJECT_PATH','..')
def predict():

    model_filename = sorted(os.listdir(f'{paths}/data/models'))[-1]
    with open(f'{paths}/data/models/{model_filename}', 'rb') as file:
        model = dill.load(file)

    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    files_list = os.listdir(f'{paths}/data/test')

    for filename in files_list:
        with open(f'{paths}/data/test/{filename}') as fin:
            form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        prediction = model.predict(df)
        dict_pred = {'car_id': df.id, 'pred': prediction}
        data = pd.DataFrame(dict_pred)
        df_pred = pd.concat([df_pred, data], axis=0)

    df_pred.to_csv(f'{paths}/data/predictions/pred{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)

if __name__ == '__main__':
    predict()
