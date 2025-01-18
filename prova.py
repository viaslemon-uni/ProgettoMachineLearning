import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import log_loss, f1_score, roc_auc_score, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
# from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb

path = "C:\\Users\\giuse\\Downloads\\ML-CUP24-"
train = pd.read_csv(path + "TR.csv", header=7)
test = pd.read_csv(path + "TS.csv", header=7)

columns = ["id", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
           "x8", "x9", "x10", "x11", "x12", "y1", "y2", "y3"]
elemento = train.columns.str.split()
elemento = elemento.map(lambda x: [float(i) for i in x])
c = np.zeros(len(elemento))

for a in range(len(elemento)):
    b = elemento[a]
    c[a] = b[0]

elemento = c

train.columns = columns
train.loc[len(train)] = elemento

elemento = test.columns.str.split()
elemento = elemento.map(lambda x: [float(i) for i in x])
c = np.zeros(len(elemento))

for a in range(len(elemento)):
    b = elemento[a]
    c[a] = b[0]

elemento = c

test.columns = columns[:13]
test.loc[len(test)] = elemento

train = train.drop(columns=["id"])
test = test.drop(columns=["id"])

trainLabel = train[["y1", "y2", "y3"]]
train = train.drop(columns=["y1", "y2", "y3"])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sur = ax.scatter(trainLabel['y1'],trainLabel['y2'],trainLabel['y3'])
pred = pd.read_csv("predNeuralNet.csv", header=None)
pred.columns = ["y1", "y2", "y3"]
sur = ax.scatter(pred['y1'],pred['y2'],pred['y3'], c='g')
pred2 = pd.read_csv("predRandomFR.csv", header=None)
pred2.columns = ["y1", "y2", "y3"]
sur = ax.scatter(pred2['y1'],pred2['y2'],pred2['y3'], c='r')
plt.show()