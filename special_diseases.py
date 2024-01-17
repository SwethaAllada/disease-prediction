import pandas as pd
import numpy as np  


class Alzheimer():
    def alzheimer():
        alzheimer = pd.read_csv("alzheimer.csv")
        healthy = alzheimer[alzheimer["Group"] != "Demented"]
        del healthy["Group"]
        average = healthy.mean()
        return average

class Diabetes():
    def diabetes():
        diab = pd.read_csv("diabetes.csv")
        healthy = diab[diab["Outcome"] == 0]
        del healthy["Outcome"]
        average = healthy.mean()
        return average

class Heart():
    def heart():
        heart = pd.read_csv("heart_disease_data.csv")
        healthy = heart[heart["target"] == 0]
        healthy = healthy.drop(["target"],axis=1)
        average = healthy.mean()
        return average


def main():
    pass

if __name__ == '__main__':
    main()