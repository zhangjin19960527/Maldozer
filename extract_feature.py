import os 
from androguard.misc import AnalyzeAPK

def JudgeFileExist(FilePath):
    '''
        给定文件路径,判断是否存在该文件
    '''
    if os.path.exists(FilePath)==True:
        return True
    else:
        return False

def ListFile(FilePath,extensions):
    '''
    给定文件夹的路径和要提取的文件扩展名,返回一个文件列表
    '''
    Files = []
    filenames = os.listdir(FilePath)
    for file in filenames:
        Absolutepath = os.path.abspath(os.path.join(FilePath,file))  # 文件的绝对路径
        if os.path.splitext(file)[1]==extensions:  # os.path.splitext分离文件名和扩展名
            Files.append(Absolutepath)
    return Files


def extract_feature(ApkDirectoryPaths):
    '''
        将给定的恶意软件目录的apk文件和良性软件的apk文件的feature提取出来
    '''

    ApkFileList = []
    for FilePath in ApkDirectoryPaths.keys():
        #将没有后缀名的和后缀名为apk的文件添加到路径中
        ApkFileList.extend(ListFile(FilePath,""))
        ApkFileList.extend(ListFile(FilePath,".apk"))

    for ApkFile in ApkFileList:
        # 将提取的apk文件的特征放入后缀名为.feature的文件中
        path = os.path.join(ApkDirectoryPaths[os.path.split(ApkFile)[0]],os.path.split(ApkFile)[1])

        if JudgeFileExist(path+'.feature'):
            pass
        else:
            try:
                a,d,dx = AnalyzeAPK(ApkFile)
                fp = open(path+'.feature','w')
                for Apkclass in dx.get_classes():
                    for meth in dx.classes[Apkclass.name].get_methods():
                        for _,call,_ in meth.get_xref_to():
                            fp.write("{}:{}\n".format(call.class_name,call.name))
                        #fp.write("{}\n".format(call.class_name))
                fp.close()
            except:
                continue
    return 


