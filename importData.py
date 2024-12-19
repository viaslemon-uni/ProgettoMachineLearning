import pandas as pd
from sklearn.preprocessing import StandardScaler

class importData:

    def __init__(self, testPath, trainPath, tensor=True):
        _test = pd.read_csv(testPath, sep='\s', header=None)
        _train = pd.read_csv(trainPath, sep='\s', header=None)
        _scaler = StandardScaler()
        _scaler = _scaler.fit(_train.iloc[:, 1:-1])
        _trasfTrain = _scaler.transform(_train.iloc[:, 1:-1])
        _trasfTest = _scaler.transform(_test.iloc[:, 1:-1])
        
        if tensor:
            import torch
            _trasfTest = torch.tensor(_trasfTest, dtype=torch.float32)
            _trasfTrain = torch.tensor(_trasfTrain, dtype=torch.float32)

        self.labelTrain = _train.iloc[:, 0]
        self.featuresTrain = _trasfTrain[:, 1:-1]
        self.featuresTest = _trasfTest[:, 1:-1]
        self.labelTest = _test.iloc[:, 0]
        self.idTrain = _train.iloc[:, -1]
        self.idTest = _test.iloc[:, -1]

if __name__ == '__main__':
    datasetPath = "C:\\Users\\giuse\\Downloads\\monk\\monks-"
    dataset = importData(datasetPath+"1.train", datasetPath+"1.test", tensor=False)
    print(dataset.labelTrain)