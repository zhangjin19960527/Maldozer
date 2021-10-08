from set_constant import apis_path,useful_api_class,mapping_to_identifier_path
from set_constant import TYPE,TYPE_list
import os 
import pickle
def main():
    '''
        官方类列表
    '''
    AuthorityApi = []
#     confoundapi=['La/','Lc/','Lb/', 'Lc/', 'Ld/', 'Le/', 'Lf/', 'Lg/', 'Lh/', 'Li/', 'Lj/', 'Lk/', 'Ll/', 'Lm/', 'Ln/', 'Lo/', 'Lp/', 'Lq/', 'Lr/', 'Ls/', 'Lt/', 'Lu/', 'Lv/', 'Lw/', 'Lx/', 'Ly/', 'Lz/', '[Lb/', '[Lc/', '[Ld/', '[Le/', '[Lf/', '[Lg/', '[Lh/', '[Li/', '[Lj/', '[Lk/', '[Ll/', '[Lm/', '[Ln/', '[Lo/', '[Lp/', '[Lq/', '[Lr/', '[Ls/', '[Lt/', 
# '[Lu/', '[Lv/', '[Lw/', '[Lx/', '[Ly/', '[Lz/','Lorg/a/','Lnet/a/','[Lcom/a/','Lcom/a/','Lio/a/']
    ThirdApi = {}
    fp = open(useful_api_class,'r')  # 读取官方class
    apilist = fp.readlines()
    for line in apilist:
        line = line.strip()
        AuthorityApi.append(line)
    fp.close()

    '''
        获得api交集
    '''
    for i in range(TYPE):
        files = os.listdir(apis_path+'/'+TYPE_list[i])
        for name in files:
            apiset = set()
            fp = open(apis_path+'/'+TYPE_list[i]+'/'+name,'r',encoding='utf-8')
            apilist = fp.readlines()
            for line in apilist:
                # if line[:3] in confoundapi or line[:5] in confoundapi:
                #     continue
                try:
                    index = line.index(';')
                except:
                    continue
                m = line[:index]
                if m not in AuthorityApi:
                    apiset.add(m)
            for tempapi in apiset:
                if tempapi in ThirdApi.keys():
                    ThirdApi[tempapi] +=1
                else:
                    ThirdApi[tempapi]=1
            fp.close()
    dict_file = open('all_third_dict.pickle','wb')
    pickle.dump(ThirdApi,dict_file)
    dict_file.close()

if __name__=='__main__':
    main()


