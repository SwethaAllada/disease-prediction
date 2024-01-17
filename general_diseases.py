import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# data = pd.read_csv("venv\data\GeneralTraining.csv")
# diseases = 

# def all_diseases():
#     data = pd.read_csv("venv\data\GeneralTraining.csv")
#     diseases = data["prognosis"].unique()
#     return diseases

data = pd.read_csv("GeneralTraining.csv")


class find_diseases():
    def diseases():
        diseases = data["prognosis"].unique()
        return diseases

class Diagnose():
    def diagnose(disease):
        ThisDisease = data[data["prognosis"] == disease]
        FeaturesForDisease = []
        for i in ThisDisease.columns:
            if len(ThisDisease[i].value_counts()) >= 2:
                FeaturesForDisease.append(i)
        return FeaturesForDisease
    
    def Prediction(disease,features, input_data):
        columns = features
        columns.append('prognosis')
        DataNeeded = data[columns]

        DataForDisease = DataNeeded[DataNeeded["prognosis"] == disease]
        DataForDisease["prognosis"] = 1
        DataWithoutDisease = DataNeeded[DataNeeded["prognosis"] != disease].sample(n = DataForDisease.shape[0])
        DataWithoutDisease["prognosis"] = 0


        n1 = DataForDisease.shape[0]
        FinalData = pd.concat([DataForDisease,DataWithoutDisease]).sample(n = 2*n1)

        x = FinalData.drop(["prognosis"],axis=1)
        y = FinalData["prognosis"]

        model = RandomForestClassifier()
        model.fit(x,y)

        input = np.array(input_data)
        input = input.reshape(1,-1)
        result = model.predict(input)


        # return FinalData,DataForDisease,DataWithoutDisease,DataNeeded,columns
        return result

def main():
    pass
    # fd = find_diseases()
    # return fd.diseases()



if __name__ == "__main__":
    main()