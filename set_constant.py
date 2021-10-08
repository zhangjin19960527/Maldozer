########## choose path ##########
apk_path = '../data/apk'
if apk_path == '../data/apk_2_152':
	TYPE = 2 # 分类数
	TYPE_list=["goodware","malware"]
elif apk_path  == '/data/ugstudent1/thordroid/code/data/apk_20':
	TYPE = 21
	TYPE_list=["goodware","Adrd","FakeInstaller","Iconosys","SendPay","BaseBridge","FakeRun","Imlog","SMSreg","DroidDream","Gappusin","Kmin","DroidKungFu","Geinimi","MobileTx","ExploitLinuxLotoor","GinMaster","Opfake","FakeDoc","Glodream","Plankton"]
elif apk_path == '../data/apk':
	TYPE = 2 # 分类数
	TYPE_list=["goodware","malware"]

apis_path='../data/apis'
identifiers_path='../data/identifiers'
train_path="../data/identifiers/train"
test_path="../data/identifiers/test"
useful_api_class="../../useful_api_class"
classes="../../classes"
mapping_to_identifier_path='../method_dict.pickle'
word2vec_model_path='./word2vec.model'
save_model_path = '../deep_learning.model' # 将训练出的模型保存到这儿
type_map={TYPE_list[i]:i for i in range(TYPE)}
########## choose path ##########


########## word2vec ##########
K = 64 # K
L = 2500 # L
########## word2vec ##########


########## CNN ##########
maxpooling_size = (14,14)
batch_size=10  # 每次喂给模型的样本数量
filter_count = 512  # filter count, 论文里是512
kernel_size_ = 3	 # kernel size 即 filter size, 论文里是3
first_neuron_count = 256	# 第一个全连接层的神经元个数 论文中是256
dropout = 0.5   # 正s则化参数 论文中是0.5
epochs_=15	   # epoch
val_split=0.2	# 验证集的比例
test_split = 0.1
KFCV = True
KFCV_K = 5
########## CNN ##########


########## fuctions ##########
def Get_file_line(filename,L=-1):#将文件的行组成list,L标识文件读多少行，当-1时全读,如果不够补0
	with open(filename,encoding='utf-8') as f:
		Sequence = f.readlines()
		if L != -1:
			lens=len(Sequence)
			if L <= lens:
				Sequence=Sequence[:L]
			else:
				Sequence.extend(['0\n']*(L-lens))
	return Sequence

def sentences_append(sentences,path,L=-1):#将path目录下的所有文件的行放入文件，形成了二维矩阵，ij中i是文件，j是文件的行,L标识每个文件读多少行，当-1时全读
	import os
	files=os.listdir(path)
	for i in range(len(files)):
		sentences.append(Get_file_line(path+'/' + files[i],L))
	return files

def read_dict(path):#读取映射到标识符的函数
	import pickle
	dict_file=open(path,'rb')
	dic=pickle.load(dict_file)
	return dic

def mkdir(path):#创建文件夹
	import os
	folder = os.path.exists(path)
 
	if not folder:				   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)			#makedirs 创建文件时如果路径不存在会创建这个路径
	else:
		print ("---  There is this folder.  ---")

def folders_set():
	mkdir(identifiers_path)
	for i in range(TYPE):
		mkdir(apis_path+'/'+TYPE_list[i])
		mkdir(train_path+'/'+TYPE_list[i])
		mkdir(test_path+'/'+TYPE_list[i])

	return True
########## fuctions ##########


########## ??? ##########
folders_set()
########## ??? ##########
