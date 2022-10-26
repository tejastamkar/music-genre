import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sklearn.model_selection as skms
from sklearn.linear_model import LogisticRegression
import sklearn.ensemble as ske
from sklearn import preprocessing
import catboost as cb
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier as knn
import pickle
from eli5.sklearn import PermutationImportance
seed = 12
np.random.seed(seed)

#Load dataset
df = pd.read_csv('src/features_3_sec.csv')
df = df.drop(['harmony_mean','harmony_var'], axis = 1)
df.label.value_counts().reset_index()

label_index = dict()
index_label = dict()
for i, x in enumerate(df.label.unique()):
    label_index[x] = i
    index_label[i] = x

df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

# remove irrelevant columns
df_shuffle.drop(['filename', 'length'], axis=1, inplace=True)
df_y = df_shuffle.pop('label')
df_X = df_shuffle

# split into train dev and test
X_train, df_test_valid_X, y_train, df_test_valid_y = skms.train_test_split(df_X, df_y, train_size=0.7, random_state=seed, stratify=df_y)
X_dev, X_test, y_dev, y_test = skms.train_test_split(df_test_valid_X, df_test_valid_y, train_size=0.66, random_state=seed, stratify=df_test_valid_y)


#Scale the features
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)  # type: ignore
X_dev = pd.DataFrame(scaler.transform(X_dev), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

pickle.dump(scaler, open('src/pickle/scalar.pkl','wb'))
pickle.dump(X_train, open('src/pickle/xtrain.pkl','wb'))

lr = LogisticRegression(random_state=seed)
lr.fit(X_train,y_train)

# Permutation Importance Feature Selection
perm = PermutationImportance(lr, random_state=seed).fit(X_train, y_train, n_iter=10)

perm_indices = np.argsort(perm.feature_importances_)[::-1]
perm_features = [X_dev.columns.tolist()[xx] for xx in perm_indices]


# Model Scoring using Permutation Importances
X_train_perm = X_train[perm_features[:30]]
X_train_rfe = X_train_perm

#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_rfe,y_train)
pickle.dump(lr, open('src/pickle/lr.pkl','wb'))

#Random Forest
rfc = ske.RandomForestClassifier(random_state=seed, n_jobs=-1)
rfc.fit(X_train_rfe, y_train)
pickle.dump(rfc, open('src/pickle/rfc.pkl','wb'))


#AdaBoost
abc = ske.AdaBoostClassifier(n_estimators=150, random_state=seed)
abc.fit(X_train_rfe, y_train)
pickle.dump(abc, open('src/pickle/abc.pkl','wb'))


#Gradient Boosting
gbc = ske.GradientBoostingClassifier(n_estimators=100, random_state=seed)
gbc.fit(X_train_rfe, y_train)
pickle.dump(gbc, open('src/pickle/gbc.pkl','wb'))


#XGBoost
xgbc = xgb.XGBClassifier(n_estimators=100, random_state=seed)
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
xgbc.fit(X_train_rfe, y_train)
pickle.dump(xgbc, open('src/pickle/xgbc.pkl','wb'))


#CatBoost
cbc = cb.CatBoostClassifier(random_state=seed, verbose=0, eval_metric='Accuracy', loss_function='MultiClass')
cbc.fit(X_train_rfe, y_train)
pickle.dump(cbc, open('src/pickle/cbc.pkl','wb'))


#KNN
cls = knn() #random_state=seed)
cls.fit(X_train_rfe, y_train)
pickle.dump(cls, open('src/pickle/cls.pkl','wb'))



