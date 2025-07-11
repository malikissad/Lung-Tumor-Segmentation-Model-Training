from src.preprocessing import preprocessing
from src.train import Train
if __name__ == "__main__":
  # preprocessing("data/datasets/Lung cancer segmentation dataset with Lung-RADS class/Lung cancer segmentation dataset with Lung-RADS class/lung_cancer_train.pkl")
  Train("data/pre-procecing/X_train.npy","data/pre-procecing/y_train.npy","data/pre-procecing/X_test.npy","data/pre-procecing/y_test.npy")