import one_get_api as oga
import two_mapping_to_identifier as tmti
import three_word2vec as tw
import four_deep_learning as fdl 
from set_constant import L,apk_path,apis_path,TYPE_list,TYPE,type_map,mapping_to_identifier_path,read_dict
import time
 




if __name__=="__main__":

	with open('21.txt','w') as fp:
		start = time.time()
		oga.get_api(TYPE,TYPE_list)
		fp.write(str(time.time() - start))
		fp.write('\n')
		start = time.time()

		dic=read_dict(mapping_to_identifier_path)
		fp.write(str(time.time() - start))
		fp.write('\n')
		start = time.time()

		tmti.mapping_to_identifier(TYPE,TYPE_list,dic)
		fp.write(str(time.time() - start))
		fp.write('\n')
		start = time.time()

		model=tw.word2vec_model(TYPE,TYPE_list)
		fp.write(str(time.time() - start))
		fp.write('\n')
		start = time.time()

		fdl.deep_learning(TYPE,TYPE_list,type_map,model)
		fp.write(str(time.time() - start))
		fp.write('\n')



