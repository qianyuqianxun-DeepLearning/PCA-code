import numpy as np    #导入矩阵与数组计算扩展包

def BatchNormalize(X):
	mu = np.mean(X,axis=0)
	sigma = np.std(X,axis=0) #计算标准差
	for i in range(X.shape[1]):
		X[:,i] = (X[:,i] - mu[i]) / sigma[i]  #进行归一化计算
	return X,mu,sigma
#三维降二维
x = np.array([[0.9,2.4,1.2,0.5,0.3,1.8,0.5,0.3,2.5,1.3],
	[1,2.6,1.7,0.7,0.7,1.4,0.6,0.6,2.6,1.1],[1,2.5,1.1,0.6,0.6,1.3,0.5,207,208,1.0]])
x = x.T  #计算x矩阵的转置矩阵
x,mu,sigma = BatchNormalize(x)
c = np.cov(x.T) #np.cov计算矩阵X的协方差
w,v = np.linalg.eig(c) #协方差矩阵的特征值和特征向量
v = v[:,0:2] #主成分分析取前两列
y = np.dot(x,v)
print('\n')
#print(x)
print(y)

