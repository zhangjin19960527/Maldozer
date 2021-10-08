from keras.utils import np_utils
from set_constant import sentences_append, L, K
import numpy as np
import random

def get_onetype(path, model, type_tag=0):
	sentences=[]
	names=sentences_append(sentences,path, L)
	y_=[type_tag]*len(names)
	x_=[]
	count = 0
	for i in sentences:
		x=[]
		for j in i:
			x.extend(model[j])
		x_.extend(x)
		count += 1
	return x_,y_,len(names)

#用于获得path文件下的所有apk的特征矩阵，以及其分类
def get_apks_and_types (path,TYPE,TYPE_list,type_map,model):
	all_x = []
	all_y = []
	all_apk_count = 0
	for i in range(TYPE):
		x, y, apk_count_of_this_type = get_onetype(path+'/'+TYPE_list[i],model,type_map[TYPE_list[i]])
		all_x.extend(x)
		all_y.extend(y)
		all_apk_count += apk_count_of_this_type
	return np.array(all_x).reshape((all_apk_count, L, K, 1)).astype('float32') / 255, np_utils.to_categorical(np.array(all_y).reshape((all_apk_count, 1)), TYPE), all_apk_count


def my_generator(x, y, start_ends:list, batch_size):
	X=[]
	Y=[]
	count = 0
	while True:
		for start_end in start_ends:
			start, end = start_end[0], start_end[1]
			for i in range(start, end):
				count += 1
				X.append(x[i])
				Y.append(y[i])
				if count == batch_size:
					res = (np.array(X), np.array(Y))
					# print(type(res[0]), type(res[1]))
					# print('hereeeeeeeeeeeeeeeeeeeeeeeeeeee',res[0].shape, res[1].shape)
					yield(res)
					# yield(np.array(X).reshape((batch_size, L, K, 1)).astype('float32') / 255,np_utils.to_categorical(Y.reshape((batch_size, 1)), TYPE))
					X = []
					Y = []
					count = 0
# index of k-fold crossing validation 
def KFCV_index(k, n):
	res = []
	step = n // k
	for i in range(k - 1):
		# train_index, val_index, train_count, val_count
		res.append(([[0,i * step], [(i + 1) * step, n]], [[i * step, (i + 1) * step]], n - step, step))
	res.append(([[0,(k - 1) * step]], [[(k - 1) * step, n]], step * (k - 1), n - step * (k - 1)))
	return res
