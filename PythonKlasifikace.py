import re
from tracemalloc import is_tracing
from unittest import main
import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import *
import pathlib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier


def train():
    global classifier
    global IsTrain

    BinaryAccuracyTrain.repaint()
    BinaryAccuracyTest.repaint()
    Console.append("Trénuju")

    #načtení dat se souboru csv
    UmisteniSouboru = UmisteniDatTextEdit.toPlainText()
    data= pd.read_csv(UmisteniSouboru.__str__() + "\\train.csv", sep = ';')
    if 'fnlwgt' in data:
        data.pop('fnlwgt')

    data = data[~data["workclass"].isin(['?'])]
    data = data[~data["occupation"].isin(['?'])]
    data = data[~data["native-country"].isin(['?'])]
    data['income'] = data['income'].replace(['<=50K.'],'<=50K')
    data['income'] = data['income'].replace(['>50K.'],'>50K')
    data['workclass'].replace('', np.nan, inplace=True)
    data.dropna(subset=['workclass'], inplace=True)
    Console.append("Data se úspěšně načetla")

    #vypsni informaci o datech
    pocetVzorku = data.shape[0]
    pocetVzorkuNad50K = data[data['income'] == '>50K'].shape[0]
    pocetVzorkuPod50K = data[data['income'] == '<=50K'].shape[0]
    Console.append("Počet vzorků: {}".format(pocetVzorku))
    Console.append("Počet lidí s platem nad $50,000: {}".format(pocetVzorkuNad50K))
    Console.append("Počet lidí s platem nejvýše $50,000: {}".format(pocetVzorkuPod50K))

    #rozděl data na vstupy a vystupy
    vystupNonBin = data['income']
    vstupy = data.drop('income', axis = 1)

    skewed= ['capital-gain', 'capital-loss']
    vstupyTransform= pd.DataFrame(data=vstupy)
    vstupyTransform[skewed]= vstupy[skewed].apply(lambda x: np.log(x+1))


    scaler =StandardScaler()               
    numerical= ['age','education-num','capital-gain','capital-loss','hours-per-week']
    vstupyTransformSt = pd.DataFrame(data = vstupyTransform)
    vstupyTransformSt[numerical] = scaler.fit_transform(vstupyTransform[numerical])
  
    #vytovoří kategorie
    vstupyEnd = pd.get_dummies(vstupyTransformSt)

    #Zakodoj nečísené vstupy
    VystupBin = vystupNonBin.map({'<=50K':0,'>50K':1})
    global dummies
    enkoder = list(vstupyEnd.columns)
    dummies = vstupyEnd.columns

    #Rozděl trenovací data na testotovací a trénovací 
    xTrain, xTest, yTrain, yTest = train_test_split(vstupyEnd,VystupBin,test_size = 0.15) 

    #Vypsání rozdělení 
    Console.append("Počet vzorků trénovacích dat: {}".format(xTrain.shape[0]))
    Console.append("Počet vzorků testovacích dat: {}".format(xTest.shape[0]))

    #Fitting classifier
    classifier= KNeighborsClassifier(n_neighbors=5, metric= 'euclidean')
    classifier.fit(xTrain, yTrain)

    #Predicting the Test Set Result 
    yPredikce = classifier.predict(xTest)

    #Výpočet přesnosti
    MaticeZamen = confusion_matrix(yTest,yPredikce)
    TP = int(MaticeZamen[0,0])
    FP = int(MaticeZamen[1,0])
    FN = int(MaticeZamen[0,1])
    TN = int(MaticeZamen[1,1])
    presnostBin = (TP + TN)/ (TP + FP + TN + FN)
    BinaryAccuracyTrain.setText("Binární přesnost-train: " + round(presnostBin,5).__str__())
    IsTrain = True
    Console.append("Hotovo")


def valid():
    global classifier
    global IsTrain

    #ověření trénování
    if IsTrain is False:
        Console.append("Není natrenována, nejdřín natrénuj")
        return 0

    Console.append("Testuju")

    UmisteniSouboru = UmisteniDatTextEdit.toPlainText()
    BinaryAccuracyTest.repaint()
    #načtení dat se souboru csv
    data= pd.read_csv(UmisteniSouboru.__str__() + "\\valid.csv", sep = ';')
    if 'fnlwgt' in data:
        data.pop('fnlwgt')

    data = data[~data["workclass"].isin(['?'])]
    data = data[~data["occupation"].isin(['?'])]
    data = data[~data["native-country"].isin(['?'])]
    data['income'] = data['income'].replace(['<=50K.'],'<=50K')
    data['income'] = data['income'].replace(['>50K.'],'>50K')
    data['workclass'].replace('', np.nan, inplace=True)
    data.dropna(subset=['workclass'], inplace=True)
    Console.append("Data se úspěšně načetla")

    #Rozdělení dat 
    vystupNonBin = data['income']
    vstupy = data.drop('income', axis = 1)

    #trosformovaní vstupů 
    skewed= ['capital-gain', 'capital-loss']
    vstupyTransform= pd.DataFrame(data=vstupy)
    vstupyTransform[skewed]= vstupy[skewed].apply(lambda x: np.log(x+1))

    #initialize a scaler, then apply it to the features
    scaler = StandardScaler()              
    numerical= ['age','education-num','capital-gain','capital-loss','hours-per-week']
    vstupyTransformSt = pd.DataFrame(data = vstupyTransform)
    vstupyTransformSt[numerical] = scaler.fit_transform(vstupyTransform[numerical])
    vstupyEnd = pd.get_dummies(vstupyTransformSt)
    vstupyEnd = vstupyEnd.reindex(columns = dummies, fill_value = 0)
    VystupBin = vystupNonBin.map({('<=50K'):0,('>50K'):1})
    enkoder = list(vstupyEnd.columns)

    xTest, yTest = vstupyEnd, VystupBin
    #Predikování validačního setu dat
    yPredikce = classifier.predict(xTest)

    #Výpočet přesnosti
    MaticeZamen = confusion_matrix(yTest,yPredikce)
    TP = int(MaticeZamen[0,0])
    FP = int(MaticeZamen[1,0])
    FN = int(MaticeZamen[0,1])
    TN = int(MaticeZamen[1,1])
    presnostBin = (TP + TN)/ (TP + FP + TN + FN)
    BinaryAccuracyTest.setText("Binární přesnost-valid: " + round(presnostBin,5).__str__())
    Console.append("Hotovo")

if __name__ == "__main__":
    global IsTrain
    IsTrain = False
    #GUI
    UmisteniSouboru = pathlib.Path().resolve()
    app = QApplication(sys.argv)
    window = QWidget()
    window.resize(300, 100)
    window.setWindowTitle('Python Klasifikace')
    layout = QVBoxLayout()

    UmisteniDatTextEdit = QTextEdit(window)
    UmisteniDatTextEdit.setText(UmisteniSouboru.__str__())
    layout.addWidget(UmisteniDatTextEdit)

    TrainButton = QPushButton('Train')
    TrainButton.clicked.connect(train)
    layout.addWidget(TrainButton)

    TestButton = QPushButton('Valid')
    TestButton.clicked.connect(valid)
    layout.addWidget(TestButton)

    BinaryAccuracyTrain = QTextBrowser(window)
    BinaryAccuracyTrain.setText("Binární přesnost-train:")
    layout.addWidget(BinaryAccuracyTrain)

    BinaryAccuracyTest = QTextBrowser(window)
    BinaryAccuracyTest.setText("Binární přesnost-valid:")
    layout.addWidget(BinaryAccuracyTest)
   
    Console = QTextBrowser(window)
    Console.setText("Console:")
    layout.addWidget(Console)

    window.setLayout(layout)
    window.show()
    app.exec()