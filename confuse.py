from keras.models import load_model
import numpy as np

def confuse(save_model_path, x_test, y_real, TYPE_list):
	model = load_model(save_model_path)
	y_pred = model.predict_classes(x_test) 

	y_real = [np.argmax(i) for i in  y_real]

	l = len(TYPE_list)
	test_l = len(x_test)
	matrix = [[0] * l for i in range(l)]

	for i in range(test_l):
		matrix[y_pred[i]][y_real[i]] += 1

	print('pred\\real', end='\t')
	for i in range(l):
		print('real ' + TYPE_list[i], end='\t')
	print()
	for i in range(l):
		print('pred ' + TYPE_list[i], end='\t')
		for j in range(l):
			print(matrix[i][j], end='\t\t')
		print()
	correct = 0
	for i in range(l):
		correct += matrix[i][i]
	print('acc: ', correct / test_l)

