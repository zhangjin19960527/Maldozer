import argparse
import os
from extract_feature import extract_feature
from set_constant import L,apk_path,apis_path
#设置L！！

def main(Args):
    #将路径都转换为绝对路径
    MalDir = os.path.abspath(Args.maldir)
    GoodDir = os.path.abspath(Args.gooddir)
    goodfeaturedir = os.path.abspath(Args.goodfeaturedir)
    malfeaturedir = os.path.abspath(Args.malfeaturedir) 
    Dir = dict()  
    Dir[MalDir] = malfeaturedir
    Dir[GoodDir] = goodfeaturedir
    extract_feature(Dir)  #将文件目录和特征目录形成一个字典,键是文件目录,值是特征目录


def ParseArgs(TYPE,TYPE_list): # 运行时添加的参数
    '''
        这里未来可能会改,现在先设定为二分类
    '''
    Args = argparse.ArgumentParser("maldozer")
    Args.add_argument("--maldir", default = apk_path+'/'+TYPE_list[1])  #训练数据的恶意样本位置
    Args.add_argument("--gooddir", default=apk_path+'/'+TYPE_list[0])  #训练数据的良性样本位置
    Args.add_argument("--goodfeaturedir",default=apis_path+'/'+TYPE_list[0])
    Args.add_argument("--malfeaturedir",default=apis_path+'/'+TYPE_list[1])
    return Args.parse_args()

def get_api(TYPE,TYPE_list):
	main(ParseArgs(TYPE,TYPE_list))
	print("Having finished first stop:get apis!")
	return

if __name__ == "__main__":
	
	from set_constant import L,TYPE_list,TYPE


	get_api(TYPE,TYPE_list)
