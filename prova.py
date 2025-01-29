import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

plt.rcParams.update({
    'font.size': 14,          # Dimensione globale del testo
    'axes.titlesize': 18,     # Titolo degli assi
    'axes.labelsize': 14,     # Etichette degli assi
    'xtick.labelsize': 12,    # Etichette dei tick sull'asse x
    'ytick.labelsize': 12,    # Etichette dei tick sull'asse y
    'legend.fontsize': 14,    # Font delle leggende
    'figure.titlesize': 20    # Titolo generale della figura
})


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

ts, val, tsLabel, valLabel = train_test_split(train, trainLabel, test_size=0.2, random_state=1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# sur = ax.scatter(trainLabel['y1'],trainLabel['y2'],trainLabel['y3'], label='train')
# ax.scatter(tsLabel['y1'],tsLabel['y2'],tsLabel['y3'], label='Training set')
ax.scatter(valLabel['y1'],valLabel['y2'],valLabel['y3'], label='Validation set')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

pred = pd.read_csv("valPredLR.csv", header=1)
pred.columns = ["y1", "y2", "y3"]
ax.scatter(pred['y1'],pred['y2'],pred['y3'], label='Linear Regression')
# pred2 = pd.read_csv("predRandomFR.csv", header=None)
# pred2.columns = ["y1", "y2", "y3"]
# sur = ax.scatter(pred2['y1'],pred2['y2'],pred2['y3'], c='r', label='Random Forest')
# pred3 = pd.read_csv("predGradientB.csv", header=None)
# pred3.columns = ["y1", "y2", "y3"]
# sur = ax.scatter(pred3['y1'],pred3['y2'],pred3['y3'], c='y', label='Gradient Boosting')
# pred4 = pd.read_csv("predAda.csv", header=None)
# pred4.columns = ["y1", "y2", "y3"]
# sur = ax.scatter(pred4['y1'],pred4['y2'],pred4['y3'], c='b', label='AdaBoost')
# predFra = pd.read_csv("C:\\Users\\giuse\\Desktop\\test_predictions.csv")
# # predFra.columns = ["y1", "y2", "y3"]
# ax.scatter(predFra['X'],predFra['Y'],predFra['Z'], c='b', label='Fra')
plt.legend()
plt.show()