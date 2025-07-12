import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
from sklearn.model_selection import train_test_split

def preprocessing(chemin):

# chargement des données
  data = pd.read_pickle(chemin)

  print(type(data))
  print(data.shape)


#masque et image
  masks = data['mask']
  images_net = data['hu_array']

#transformer en tableau NumPy + canal

  X = np.stack(images_net.values)[...,np.newaxis]
  Y = np.stack(masks.values)[...,np.newaxis]

  print(X.shape)
  print(Y.shape)

#resize
  X_resize = tf.image.resize(X,[256,256]).numpy()
  Y_resize = tf.image.resize(Y,[256,256],method='nearest').numpy()

#canal 3 recommender pour swin-unit
  X_canal = np.repeat(X_resize,3,axis=-1)

# # Normaliser uniquement les images
#   X_norm = (X_canal - X_canal.min()) / (X_canal.max() - X_canal.min())

# # S'assurer que les masques sont bien binaires
#   Y_bin = (Y_resize > 0).astype(np.uint8)

#   print(X_norm)
#   print(Y_bin)

  X_norm = np.zeros_like(X_canal)
  
  for i in range(X_canal.shape[0]):
      img = X_canal[i]
      img_min = img.min()
      img_max = img.max()
      if img_max > img_min:  # Éviter division par zéro
        X_norm[i] = (img - img_min) / (img_max - img_min)
      else:
            X_norm[i] = img
    
  # ✅ CORRECTION 2: Masques binaires avec meilleure gestion
  Y_bin = (Y_resize > 0.5).astype(np.float32)  # float32 pour PyTorch


  X_train, X_test, y_train, y_test = train_test_split(X_norm, Y_bin, test_size=0.2, random_state=42)

  np.save("data/pre-procecing/X_train.npy",X_train)
  np.save("data/pre-procecing/X_test.npy",X_test)
  np.save("data/pre-procecing/y_train.npy",y_train)
  np.save("data/pre-procecing/y_test.npy",y_test)