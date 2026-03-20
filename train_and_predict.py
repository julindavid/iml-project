import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score


def apply_grouped_mapping(df, column, grouping_dict):
    flat_map = {idx: group_name for group_name, id_list in grouping_dict.items() for idx in id_list}
    return df[column].map(flat_map).fillna('Other')


def preprocess(df, is_train=True):
    df = df.copy()
    # Basic fills
    if 'A1Cresult' in df.columns:
        df['A1Cresult'] = df['A1Cresult'].fillna('none')
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum'] = df['max_glu_serum'].fillna('none')
    if 'race' in df.columns:
        df['race'] = df['race'].replace('?', 'Caucasian')

    # Drop low-information / leaking columns
    drop_cols = [c for c in [
        'weight', 'payer_code', 'medical_specialty', 'encounter_id',
        'examide', 'troglitazone', 'citoglipton', 'diag_1', 'diag_2', 'diag_3'
    ] if c in df.columns]
    df = df.drop(columns=drop_cols)

    admission_type_groups = {
        'Emergency': [1, 7], 'Urgent': [2], 'Elective': [3], 'Newborn': [4], 'Unknown': [5, 6, 8]
    }
    discharge_groups = {
        'Home': [1, 6, 8],
        'Expired': [11, 19, 20, 21],
        'Transferred': [2, 3, 4, 5, 10, 12, 13, 14, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30],
        'Unknown': [18, 25, 26]
    }
    admission_source_groups = {
        'Referral': [1, 2, 3],
        'Transfer': [4, 5, 6, 10, 18, 19, 22, 25, 26],
        'Emergency': [7], 'Birth': [11, 12, 13, 14, 23, 24],
        'Unknown': [9, 15, 17, 20, 21], 'Other': [8]
    }

    if 'admission_type_id' in df.columns:
        df['admission_type'] = apply_grouped_mapping(df, 'admission_type_id', admission_type_groups)
    if 'discharge_disposition_id' in df.columns:
        df['discharge_disposition'] = apply_grouped_mapping(df, 'discharge_disposition_id', discharge_groups)
    if 'admission_source_id' in df.columns:
        df['admission_source'] = apply_grouped_mapping(df, 'admission_source_id', admission_source_groups)

    # Map ordinal age
    if 'age' in df.columns:
        age_map = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
            '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
        }
        df['age'] = df['age'].map(age_map)

    # Binary mappings (if present)
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

    # One-hot nominal columns
    nominal_cols = [
        'race', 'gender', 'metformin', 'repaglinide', 'nateglinide',
        'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        'insulin', 'glyburide-metformin', 'max_glu_serum', 'A1Cresult',
        'admission_type', 'discharge_disposition', 'admission_source'
    ]
    cols_to_encode = [c for c in nominal_cols if c in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    # Remove id if present (keep separate externally)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    return df


def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # keep ids for submission
    if 'id' in test.columns:
        test_ids = test['id'].astype(int)
    else:
        test_ids = test['encounter_id'] if 'encounter_id' in test.columns else pd.Series(range(len(test)))

    # target mapping (train.csv has 'readmitted' with strings)
    if 'readmitted' in train.columns:
        train['readmitted'] = train['readmitted'].map({'No':0,'<30':1,'>30':2})

    y = train['readmitted'] if 'readmitted' in train.columns else None

    X = preprocess(train.drop(columns=['readmitted']) if 'readmitted' in train.columns else train)
    X_test = preprocess(test)

    # align columns
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    # Model and CV
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

    print('Running 5-fold CV...')
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f'CV accuracy: {np.round(acc_scores.mean(),4)} ± {np.round(acc_scores.std(),4)}')
    print(f'CV f1_macro: {np.round(f1_scores.mean(),4)} ± {np.round(f1_scores.std(),4)}')

    # Fit on full training data
    model.fit(X, y)

    preds = model.predict(X_test)
    # map back to labels
    label_map = {0: 'No', 1: '<30', 2: '>30'}
    preds_str = [label_map.get(int(p), 'No') for p in preds]

    submission = pd.DataFrame({'id': test_ids, 'readmitted': preds_str})
    submission.to_csv('outputs/kaggle_submission.csv', index=False)
    print('Wrote outputs/kaggle_submission.csv')


if __name__ == '__main__':
    main()
