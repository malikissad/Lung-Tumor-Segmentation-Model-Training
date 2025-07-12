from src.preprocessing import preprocessing
from src.train import Train
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # preprocessing("data/datasets/Lung cancer segmentation dataset with Lung-RADS class/Lung cancer segmentation dataset with Lung-RADS class/lung_cancer_train.pkl")
  Train("data/pre-procecing/X_train.npy","data/pre-procecing/y_train.npy","data/pre-procecing/X_test.npy","data/pre-procecing/y_test.npy")
  # Y_train = np.load("data/pre-procecing/y_train.npy")
 
  # print(Y_train.shape)
  # print("Nombre de pixels '1' dans Y_train :", np.sum(Y_train == 1))
  # print("Nombre de pixels '0' dans Y_train :", np.sum(Y_train == 0))