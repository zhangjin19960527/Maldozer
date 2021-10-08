

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.utils import np_utils
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

from set_constant import sentences_append
from set_constant import train_path,test_path
from set_constant import save_model_path,L,K
from set_constant import filter_count, kernel_size_,first_neuron_count,dropout,epochs_,batch_size_,validation_split_

# x_train, x_test是 train_apk_count*(L*K) 的(二维)numpy矩阵
# 注意此处要把每个L * K的矩阵展平了成一个(L * K)的向量
# y_train, y_test是 test_apk_count*1 的(一维)numpy矩阵
def get_onetype(path,model,type=0):
    sentences=[]
    names=sentences_append(sentences,path)
    y_=[type]*len(names)
    x_=[]
    for i in sentences:
        x=[]
        for j in i:
            x.extend(model[j])
        x_.extend(x)
    return x_,y_,len(names)

#用于获得path文件下的所有apk的特征矩阵，以及其分类
def get_apks_and_types (path,TYPE,TYPE_line,type_map,model):
    apk_count=0
    X=[]
    Y=[]
    # 把所有训练集的矩阵读到这个二维张量中
    for i in range(0,TYPE):
        x,y,z=get_onetype(path+'/'+TYPE_line[i],model,type_map[TYPE_line[i]])
        x=np.array(x)
        X.extend(x)
        Y.extend(y)
        apk_count+=z
    X=np.array(X)
    X= X.reshape((apk_count, L , K, 1))
    print('X shape:', X.shape)
    #X= X.reshape(X.shape[0],L,K,1)
    Y=np.array(Y)
    Y= Y.reshape((apk_count, 1))
    print('X shape:', X.shape)
    
   
   
   
    Y = np_utils.to_categorical(Y, TYPE)
    #Y= Y.reshape((apk_count, 2,1))
    return X,Y,apk_count

def deep_learning(TYPE,TYPE_line,type_map,word2vec_model):
	
	# 训练集,测试集
	# 我不太清楚(a,b,c)和(a,b,c,1)的张量有啥区别,为了保险起见这样写,把输入矩阵调成和神经网络一样的形状
	x_train,y_train,train_apk_count = get_apks_and_types(train_path,TYPE,TYPE_line,type_map,word2vec_model)
	x_test,y_test,test_apk_count = get_apks_and_types(test_path,TYPE,TYPE_line,type_map,word2vec_model)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	#print('x_train shape:', x_train.shape)
	#print('x_test shape:', x_test.shape)

	# 神经网络
	model = Sequential()
	# 卷积
	model.add(layers.Conv2D(filters=filter_count, kernel_size=kernel_size_, activation='relu', input_shape=(L , K, 1)))
	model.summary()
	# 池化
	model.add(layers.MaxPooling2D())
	model.summary()
	# 第一个全连接
	model.add(layers.Dense(units=first_neuron_count, activation='relu'))
	model.summary()
	# 正则化
	model.add(layers.Dropout(dropout))
	model.add(layers.Flatten())
	# 第二个全连接
	model.add(layers.Dense(units=TYPE, activation='softmax'))
	model.summary()

	model.compile(optimizer=RMSprop(lr=1e-4),
			loss='binary_crossentropy',
				  metrics=['acc'])

	history = model.fit(x_train, y_train,
						epochs=epochs_,
						batch_size=batch_size_,
						validation_split=validation_split_)

	model.save(save_model_path)

	# 根据结果画图
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(acc))

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()

	plt.figure()

	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	
	# 测试集
	test_loss, test_acc = model.evaluate(x_test, y_test)
	print('Testing and accuracy:', test_acc)
	
	plt.show()

	
	
	print("Having finished fourth stop:deep learning!")

if __name__ == "__main__":

	from gensim.models import Word2Vec
	from set_constant import TYPE,TYPE_list,type_map
	from set_constant import word2vec_model_path


	word2vec_model = Word2Vec.load(word2vec_model_path)
	deep_learning(TYPE,TYPE_list,type_map,word2vec_model)
