import os
import random
from set_constant import read_dict,Get_file_line
import multiprocessing

def fun(id,val_split,apis_path,TYPE_list,dic,files,train_path,test_path):
    
    num = len(files)
    test_number = int(num*val_split)
    train_number = num - test_number

    for j in range(num):
        f= Get_file_line(apis_path+'/'+TYPE_list+'/'+files[j])
        lens = len(f)
        if lens==0:
            continue
        g = []
        for k in f:#限制sentence长为L
            m = k[:-1]
            if m in dic.keys():#当长度小于L时后面补0；当api不在dict关键字里面，map到0
                identifier=str(dic[m])+'\n'
            else:
                identifier='0\n'
            g.append(identifier)
        if j < train_number:#分为数据集和训练集
            with open(train_path+'/'+TYPE_list+'/'+files[j],'w') as x:
                x.writelines(g)
        else:
            with open(test_path+'/'+TYPE_list+'/'+files[j],'w') as x:
                x.writelines(g)

def process_mapping_to_identifier(TYPE,TYPE_list,dic,process_number=20):
    from set_constant import L,apis_path,train_path,test_path,val_split
    for i in range(TYPE):
        print(apis_path+'/'+TYPE_list[i])
        files=os.listdir(apis_path+'/'+TYPE_list[i])
        split_ = int(len(files)/process_number)
        random.shuffle(files)
        Processes = []
        for j in range(process_number):
            if j != process_number-1:
                process_ApkFile = files[j*split_:(j+1)*split_]
            else:
                process_ApkFile = files[j*split_:]
            p = multiprocessing.Process(target=fun,args=(j,val_split,apis_path,TYPE_list[i],dic,process_ApkFile,train_path,test_path))
            p.start()
            Processes.append(p)
        for pro in Processes:
            pro.join()
        
        


def mapping_to_identifier(TYPE,TYPE_list,dic):
    
    from set_constant import L,apis_path,train_path,test_path,val_split
    for i in range(TYPE):
        files=os.listdir(apis_path+'/'+TYPE_list[i])
        number=len(files)
        test_number=int(number*val_split)
        train_number=number-test_number
        random.shuffle(files)
        for j in range(number):
            f=Get_file_line(apis_path+'/'+TYPE_list[i]+'/'+files[j])
            lens=len(f)
            if lens==0:
                continue
            g=[]
            for k in f:#限制sentence长为L
                m = k[:-1]
                identifier=""
                if m in dic.keys():#当长度小于L时后面补0；当api不在dict关键字里面，map到0
                    identifier=str(dic[m])+'\n'
                else:
                    identifier='0\n'
                g.append(identifier)
                
            if j < train_number:#分为数据集和训练集
                with open(train_path+'/'+TYPE_list[i]+'/'+files[j],'w') as x:
                    x.writelines(g)
            else:
                with open(test_path+'/'+TYPE_list[i]+'/'+files[j],'w') as x:
                    x.writelines(g)
    return 


if  __name__ == "__main__":
    from set_constant import TYPE,TYPE_list,mapping_to_identifier_path
    
    dic=read_dict(mapping_to_identifier_path)
    #mapping_to_identifier(TYPE,TYPE_list,dic)
    process_mapping_to_identifier(TYPE,TYPE_list,dic)
