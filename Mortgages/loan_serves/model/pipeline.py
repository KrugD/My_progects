import datetime

import dill
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def filter_data(df):
    df_new = df.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    df_new = df_new.drop(columns_to_drop, axis=1)

    return df_new


def delet_outliers_year(df):
    df_new = df.copy()
    q25 = df_new['year'].quantile(0.25)
    q75 = df_new['year'].quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
    df_new.loc[df_new['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df_new.loc[df_new['year'] > boundaries[1], 'year'] = round(boundaries[1])

    return df_new

def add_features(df):
    def short_model(x):
        if not pd.isna(x):

            return x.lower().split(' ')[0]

        else:

            return x

    df_new = df.copy()
    df.loc[:, 'short_model'] = df['model'].apply(short_model)
    df.loc[:, 'age_category'] = df['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))

    return df_new


def main():
    print('Category_cars Prediction Pipeline')

    df = pd.read_csv('data/30.6 homework.csv')

    X = df.drop(['price_category'], axis=1)
    y = df['price_category']

    categorical_features = make_column_selector(dtype_include=['object'])
    numerical_features = make_column_selector(
        dtype_include=['int64', 'float64'])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor_1 = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outlier_year', FunctionTransformer(delet_outliers_year)),
        ('add_features', FunctionTransformer(add_features))
    ])

    preprocessor_2 = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    )

    best_score = .0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor_1', preprocessor_1),
            ('preprocessor_2', preprocessor_2),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(
            f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)

    print(
        f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    model_filename = f'price_cars.pkl'
    with open (model_filename, 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Price prediction model',
                'author': 'Denis Kruglov',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


