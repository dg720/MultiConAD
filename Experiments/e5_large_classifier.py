import os
import sys
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_language', required=True, choices=['en', 'gr', 'cha', 'spa'])
parser.add_argument('--task', required=True, choices=['binary', 'multiclass'])
parser.add_argument('--translated', required=True, choices=['yes', 'no'])
parser.add_argument('--training', required=True, choices=['mono', 'multi'])
parser.add_argument('--cache_dir', default=None, help='Directory to cache embeddings')
parser.add_argument('--svm_class_weight', choices=['balanced', 'none'], default='balanced')
args = parser.parse_args()

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CLEANED      = os.path.join(PROJECT_ROOT, "Preprocessing_text", "cleaned")
TRANSLATED   = os.path.join(PROJECT_ROOT, "Translation", "translated")
CACHE_DIR    = args.cache_dir or os.path.join(SCRIPT_DIR, "embedding_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
svm_class_weight = None if args.svm_class_weight == 'none' else 'balanced'

_model = None

def _get_model():
    global _model
    if _model is None:
        print("  Loading E5-large model...")
        _model = SentenceTransformer('intfloat/multilingual-e5-large').to(device)
    return _model


def load(split, lang, translated=False):
    if translated and lang != 'en':
        name_map = {'gr': 'greek', 'cha': 'chinese', 'spa': 'spanish'}
        path = os.path.join(TRANSLATED, f"{split}_{name_map[lang]}_translated.jsonl")
    else:
        name_map = {'en': 'english', 'gr': 'greek', 'cha': 'chinese', 'spa': 'spanish'}
        path = os.path.join(CLEANED, f"{split}_{name_map[lang]}.jsonl")
    df = pd.read_json(path, lines=True)
    if lang == 'en' and 'translated' not in df.columns:
        df['translated'] = df['Text_interviewer_participant']
    return df


def get_embeddings(df, text_col, cache_key):
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings: {cache_key}")
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(df):
            return embeddings
        print(
            f"  Cache row mismatch for {cache_key}: cache={embeddings.shape[0]} current={len(df)}. "
            "Rebuilding cache."
        )
    print(f"  Encoding {len(df)} texts: {cache_key}")
    model = _get_model()
    texts = ["passage: " + str(t) for t in df[text_col].fillna('').tolist()]
    embeddings = model.encode(texts, normalize_embeddings=True, device=device,
                              batch_size=32, show_progress_bar=True)
    np.save(cache_path, embeddings)
    return embeddings


use_translated = args.translated == 'yes'
text_col = 'translated' if use_translated else 'Text_interviewer_participant'
trans_tag = 'trans' if use_translated else 'orig'

train_en  = load('train', 'en',  use_translated)
test_en   = load('test',  'en',  use_translated)
train_gr  = load('train', 'gr',  use_translated)
test_gr   = load('test',  'gr',  use_translated)
train_cha = load('train', 'cha', use_translated)
test_cha  = load('test',  'cha', use_translated)
train_spa = load('train', 'spa', use_translated)
test_spa  = load('test',  'spa', use_translated)

all_trains = {'en': train_en, 'gr': train_gr, 'cha': train_cha, 'spa': train_spa}
all_tests  = {'en': test_en,  'gr': test_gr,  'cha': test_cha,  'spa': test_spa}
lang_map   = {'en': 'english', 'gr': 'greek', 'cha': 'chinese', 'spa': 'spanish'}

if args.training == 'mono':
    train_dfs = [all_trains[args.test_language]]
    train_tag = f'mono_{args.test_language}'
else:
    train_dfs = list(all_trains.values())
    train_tag = 'multi'

test_df = all_tests[args.test_language].copy()

train_combined = pd.concat(train_dfs, ignore_index=True)
train_combined['Diagnosis'] = train_combined['Diagnosis'].replace({'AD': 'Dementia'})
test_df['Diagnosis'] = test_df['Diagnosis'].replace({'AD': 'Dementia'})

if args.task == 'binary':
    train_combined = train_combined[train_combined['Diagnosis'] != 'MCI'].reset_index(drop=True)
    test_df = test_df[test_df['Diagnosis'] != 'MCI'].reset_index(drop=True)

# Cache key encodes exactly what data is being embedded
train_cache_key = f"train_{train_tag}_{args.task}_{trans_tag}"
test_cache_key  = f"test_{args.test_language}_{args.task}_{trans_tag}"

print(f"\n{'='*60}")
print(f"E5-large | training={args.training} | test={args.test_language} | task={args.task} | translated={args.translated}")
print(f"Train size: {len(train_combined)} | Test size: {len(test_df)}")
print(f"Label dist (train): {dict(train_combined['Diagnosis'].value_counts())}")
print(f"Label dist (test):  {dict(test_df['Diagnosis'].value_counts())}")
print(f"SVM class_weight: {args.svm_class_weight}")
print('='*60)

X_train = get_embeddings(train_combined, text_col, train_cache_key)
X_test  = get_embeddings(test_df,        text_col, test_cache_key)
y_train = train_combined['Diagnosis'].values
y_test  = test_df['Diagnosis'].values

classifiers = {
    'Decision Tree':       (DecisionTreeClassifier(random_state=42),                        {'max_depth': [10, 20, 30]}),
    'Random Forest':       (RandomForestClassifier(random_state=42),                         {'n_estimators': [50, 100, 200]}),
    'SVM':                 (SVC(random_state=42, class_weight=svm_class_weight),             {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000),              {'C': [0.1, 1, 10]}),
}

for name, (clf, params) in classifiers.items():
    grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(f"\n--- {name} (best params: {grid.best_params_}) ---")
    print(classification_report(y_test, y_pred))

# Explicitly release GPU memory so the next subprocess gets a clean VRAM state
if _model is not None:
    del _model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
