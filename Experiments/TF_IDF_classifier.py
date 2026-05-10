import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_language', required=True, choices=['en', 'gr', 'cha', 'spa'])
parser.add_argument('--task', required=True, choices=['binary', 'multiclass'])
parser.add_argument('--translated', required=True, choices=['yes', 'no'])
parser.add_argument('--training', required=True, choices=['mono', 'multi'])
parser.add_argument('--svm_class_weight', choices=['balanced', 'none'], default='balanced')
args = parser.parse_args()

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CLEANED      = os.path.join(PROJECT_ROOT, "Preprocessing_text", "cleaned")
TRANSLATED   = os.path.join(PROJECT_ROOT, "Translation", "translated")

def load(split, lang, translated=False):
    if translated and lang != 'en':
        name_map = {'gr': 'greek', 'cha': 'chinese', 'spa': 'spanish'}
        path = os.path.join(TRANSLATED, f"{split}_{name_map[lang]}_translated.jsonl")
    else:
        name_map = {'en': 'english', 'gr': 'greek', 'cha': 'chinese', 'spa': 'spanish'}
        path = os.path.join(CLEANED, f"{split}_{name_map[lang]}.jsonl")
    df = pd.read_json(path, lines=True)
    # English uses Text_interviewer_participant as its own "translated" field
    if lang == 'en' and 'translated' not in df.columns:
        df['translated'] = df['Text_interviewer_participant']
    return df

use_translated = args.translated == 'yes'
text_col = 'translated' if use_translated else 'Text_interviewer_participant'
svm_class_weight = None if args.svm_class_weight == 'none' else 'balanced'

train_en  = load('train', 'en',  use_translated)
test_en   = load('test',  'en',  use_translated)
train_gr  = load('train', 'gr',  use_translated)
test_gr   = load('test',  'gr',  use_translated)
train_cha = load('train', 'cha', use_translated)
test_cha  = load('test',  'cha', use_translated)
train_spa = load('train', 'spa', use_translated)
test_spa  = load('test',  'spa', use_translated)

lang_map = {'en': 'english', 'gr': 'greek', 'cha': 'chinese', 'spa': 'spanish'}

all_trains = {'en': train_en, 'gr': train_gr, 'cha': train_cha, 'spa': train_spa}
all_tests  = {'en': test_en,  'gr': test_gr,  'cha': test_cha,  'spa': test_spa}

if args.training == 'mono':
    train_dfs = [all_trains[args.test_language]]
else:
    train_dfs = list(all_trains.values())

test_df = all_tests[args.test_language]

# Normalise English Dementia labels across combined training
train_combined = pd.concat(train_dfs, ignore_index=True)
train_combined['Diagnosis'] = train_combined['Diagnosis'].replace({'AD': 'Dementia'})
test_df = test_df.copy()
test_df['Diagnosis'] = test_df['Diagnosis'].replace({'AD': 'Dementia'})

if args.task == 'binary':
    train_combined = train_combined[train_combined['Diagnosis'] != 'MCI']
    test_df = test_df[test_df['Diagnosis'] != 'MCI']

X_train = train_combined[text_col].fillna('').astype(str)
y_train = train_combined['Diagnosis']
X_test  = test_df[text_col].fillna('').astype(str)
y_test  = test_df['Diagnosis']

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

classifiers = {
    'Decision Tree':      (DecisionTreeClassifier(random_state=42),      {'max_depth': [10, 20, 30]}),
    'Random Forest':      (RandomForestClassifier(random_state=42),       {'n_estimators': [50, 100, 200]}),
    'Naive Bayes':        (MultinomialNB(),                               {'alpha': [0.5, 1.0, 1.5]}),
    'SVM':                (SVC(random_state=42, class_weight=svm_class_weight),  {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'Logistic Regression':(LogisticRegression(random_state=42, max_iter=1000), {'C': [0.1, 1, 10]}),
}

print(f"\n{'='*60}")
print(f"TF-IDF | training={args.training} | test={args.test_language} | task={args.task} | translated={args.translated}")
print(f"Train size: {len(train_combined)} | Test size: {len(test_df)}")
print(f"Train langs: {[lang_map[k] for k in all_trains if all_trains[k] is not None and any(all_trains[k].equals(d) for d in train_dfs)]}")
print(f"Label dist (train): {dict(y_train.value_counts())}")
print(f"Label dist (test):  {dict(y_test.value_counts())}")
print(f"SVM class_weight: {args.svm_class_weight}")
print('='*60)

for name, (clf, params) in classifiers.items():
    grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_tfidf, y_train)
    y_pred = grid.predict(X_test_tfidf)
    print(f"\n--- {name} (best params: {grid.best_params_}) ---")
    print(classification_report(y_test, y_pred))
