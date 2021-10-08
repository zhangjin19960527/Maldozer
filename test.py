import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

from set_constant import train_path,test_path
from set_constant import save_model_path,L,K
from set_constant import filter_count, kernel_size_,first_neuron_count,dropout,epochs_,batch_size
from set_constant import maxpooling_size, test_split, val_split, KFCV, KFCV_K
from my_generator import my_generator, get_apks_and_types, KFCV_index
from confuse import confuse
import random


def deep_learning(TYPE,TYPE_list,type_map,word2vec_model):
	
	# 训练集,测试集
	# x_train,y_train,train_apk_count = get_apks_and_types(train_path,TYPE,TYPE_list,type_map,word2vec_model)
	x_test,y_test,test_apk_count = get_apks_and_types(test_path,TYPE,TYPE_list,type_map,word2vec_model)
	confuse(save_model_path, x_test, y_test, TYPE_list)

if __name__ == "__main__":

	from gensim.models import Word2Vec
	from set_constant import TYPE,TYPE_list,type_map
	from set_constant import word2vec_model_path


	word2vec_model = Word2Vec.load(word2vec_model_path)
	deep_learning(TYPE,TYPE_list,type_map,word2vec_model)
