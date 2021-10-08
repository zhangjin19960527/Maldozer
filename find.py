import pickle
from set_constant import mapping_to_identifier_path,useful_api_class,classes
def set_dict():##非通用
	dict_file=open(mapping_to_identifier_path,'wb')
	useful_api=open(useful_api_class,'w')
	dic={}
	lines=[]
	l=0
	key=0
	with open(classes,encoding='gbk',errors='ignore') as X:
		for line in X:
			l+=1
			a=line.find('<a href="')
			b=line.rfind('</a></td>')
			if a!=-1 :
				if b!=-1:
					y=line[a+20:b]
					x=y[:y.find('">')]
					x='L'+x.replace('.','$')
					lines.append(x+'\n')
					dic[x]=key+1
					key+=1
				
				else:
					print(str(l)+':'+line)
	useful_api.writelines(lines)
	useful_api.close()
	pickle.dump(dic,dict_file)
	dict_file.close()
	
	return key



			
if __name__=="__main__":
	set_dict()
