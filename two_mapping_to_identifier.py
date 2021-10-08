import os
import random
from set_constant import read_dict,Get_file_line
def mapping_to_identifier(TYPE,TYPE_list,dic):
	
	from set_constant import L,apis_path,train_path,test_path,validation_split_
	for i in range(TYPE):
		files=os.listdir(apis_path+'/'+TYPE_list[i])
		number=len(files)
		test_number=int(number*validation_split_)
		train_number=number-test_number
		random.shuffle(files)
		for j in range(number):
			f=Get_file_line(apis_path+'/'+TYPE_list[i]+'/'+files[j])
			lens=len(f)
			g=[]
			for k in range(L):#限制sentence长为L
				if (k<lens) and (f[k][:-1] in dic.keys()):#当长度小于L时后面补0；当api不在dict关键字里面，map到0
					m=f[k][:-1]
					g.append(str(dic[m])+'\n')
				else:
					g.append('0\n')
			# if j < train_number:#分为数据集和训练集
			# 	with open(train_path+'/'+TYPE_list[i]+'/'+files[j],'w') as x:
			# 		x.writelines(g)
			# else:
			# 	with open(test_path+'/'+TYPE_list[i]+'/'+files[j],'w') as x:
			# 		x.writelines(g)

			num = random.random()
			if num < 0.8:
				with open(train_path+'/'+TYPE_list[i]+'/'+files[j],'w') as x:
					x.writelines(g)
			else:
				with open(test_path+'/'+TYPE_list[i]+'/'+files[j],'w') as x:
					x.writelines(g)
	print("Having finished second stop:mapping to idnetifier!")
	return 

if  __name__ == "__main__":
	from set_constant import TYPE,TYPE_list,mapping_to_identifier_path
	
	dic=read_dict(mapping_to_identifier_path)
	mapping_to_identifier(TYPE,TYPE_list,dic)
