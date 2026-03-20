import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import time


def apply_grouped_mapping(df, column, grouping_dict):
    flat_map = {idx: group_name for group_name, id_list in grouping_dict.items() for idx in id_list}
    return df[column].map(flat_map).fillna('Other')


def preprocess(df):
    df = df.copy()
    if 'A1Cresult' in df.columns:
        df['A1Cresult'] = df['A1Cresult'].fillna('none')
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum'] = df['max_glu_serum'].fillna('none')
    if 'race' in df.columns:
        df['race'] = df['race'].replace('?', 'Caucasian')

    drop_cols = [c for c in ['weight','payer_code','medical_specialty','encounter_id','examide','troglitazone','citoglipton','diag_1','diag_2','diag_3'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    admission_type_groups = {'Emergency':[1,7],'Urgent':[2],'Elective':[3],'Newborn':[4],'Unknown':[5,6,8]}
    discharge_groups = {'Home':[1,6,8],'Expired':[11,19,20,21],'Transferred':[2,3,4,5,10,12,13,14,15,16,17,22,23,24,27,28,29,30],'Unknown':[18,25,26]}
    admission_source_groups = {'Referral':[1,2,3],'Transfer':[4,5,6,10,18,19,22,25,26],'Emergency':[7],'Birth':[11,12,13,14,23,24],'Unknown':[9,15,17,20,21],'Other':[8]}

    if 'admission_type_id' in df.columns:
        df['admission_type'] = apply_grouped_mapping(df, 'admission_type_id', admission_type_groups)
    if 'discharge_disposition_id' in df.columns:
        df['discharge_disposition'] = apply_grouped_mapping(df, 'discharge_disposition_id', discharge_groups)
    if 'admission_source_id' in df.columns:
        df['admission_source'] = apply_grouped_mapping(df, 'admission_source_id', admission_source_groups)

    if 'age' in df.columns:
        age_map = {'[0-10)':0,'[10-20)':1,'[20-30)':2,'[30-40)':3,'[40-50)':4,'[50-60)':5,'[60-70)':6,'[70-80)':7,'[80-90)':8,'[90-100)':9}
        df['age'] = df['age'].map(age_map)

    bin_maps = {
        'tolazamide': {'No':0,'Steady':1,'Up':2},
        'acetohexamide': {'No':0,'Steady':1},
        'glimepiride-pioglitazone': {'No':0,'Steady':1},
        'metformin-pioglitazone': {'No':0,'Steady':1},
        'metformin-rosiglitazone': {'No':0,'Steady':1},
        'glipizide-metformin': {'No':0,'Steady':1},
        'tolbutamide': {'No':0,'Steady':1},
        'change': {'Ch':1,'No':0},
        'diabetesMed': {'Yes':1,'No':0}
    }
    for col, mp in bin_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mp)

    nominal_cols = ['race','gender','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','glipizide','glyburide','pioglitazone','rosiglitazone','acarbose','miglitol','insulin','glyburide-metformin','max_glu_serum','A1Cresult','admission_type','discharge_disposition','admission_source']
    cols_to_encode = [c for c in nominal_cols if c in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    return df


def run():
    train = pd.read_csv('train.csv')
    print('Train rows:', len(train))
    print('Readmitted value counts:')
    print(train['readmitted'].value_counts(normalize=False))
    print(train['readmitted'].value_counts(normalize=True))

    y = train['readmitted'].map({'No':0,'<30':1,'>30':2})
    X = preprocess(train.drop(columns=['readmitted']))

    # baseline RF CV
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    print('Running baseline CV (f1_macro)...')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    print('Baseline f1_macro:', np.round(f1_scores.mean(),4), '±', np.round(f1_scores.std(),4))

    # Quick manual tuning: try a few candidate RandomForest configs (sequential to avoid parallel spawn issues)
    candidates = [
        {'n_estimators':200,'max_depth':20,'max_features':'sqrt','min_samples_split':5,'class_weight':'balanced_subsample'},
        {'n_estimators':300,'max_depth':30,'max_features':'sqrt','min_samples_split':5,'class_weight':'balanced'},
        {'n_estimators':300,'max_depth':None,'max_features':'log2','min_samples_split':2,'class_weight':'balanced'},
        {'n_estimators':150,'max_depth':15,'max_features':0.5,'min_samples_split':5,'class_weight':'balanced_subsample'},
    ]
    best_score = -1
    best_cfg = None
    best_model = None
    for cfg in candidates:
        print('Testing cfg:', cfg)
        clf = RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=13), scoring='f1_macro', n_jobs=1)
        mean = scores.mean()
        print(' f1_macro:', np.round(mean,4), '±', np.round(scores.std(),4))
        if mean > best_score:
            best_score = mean
            best_cfg = cfg
            best_model = clf
    print('Selected best f1_macro:', np.round(best_score,4), 'with', best_cfg)
    # fit best on full data
    best_model.set_params(n_jobs=-1)
    best_model.fit(X, y)
    test = pd.read_csv('test.csv')
    ids = test['id'] if 'id' in test.columns else test['encounter_id']
    X_test = preprocess(test)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    preds = best_model.predict(X_test)
    label_map = {0:'No',1:'<30',2:'>30'}
    preds_str = [label_map.get(int(p),'No') for p in preds]
    pd.DataFrame({'id': ids, 'readmitted': preds_str}).to_csv('outputs/kaggle_submission.csv', index=False)
    print('Wrote outputs/kaggle_submission.csv with tuned model')


if __name__ == '__main__':
    run()
