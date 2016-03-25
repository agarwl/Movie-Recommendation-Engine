#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open('../datasets/ua.base','r') as f:
	train = pd.read_csv(f)
	train = train.drop('timestamp',1)


with open('../datasets/ua.test','r') as f:
	test = pd.read_csv(f)
	test = test.drop('timestamp',1)

X_test = test.values
y = test['rating'].values
X_test = np.delete(X_test,2,axis = 1)

X = train.values
users = pd.unique(train.userId.ravel())
movies = pd.unique(train.movieId.ravel())
u_len = train.userId.max()
m_len = train.movieId.max()

R = np.zeros(shape = (max(users)+1,max(movies)+1))
for entry in X:
	R[entry[0]][entry[1]] = entry[2]

user_movies, item_users, item_means, mean_ratings, sigmas = {},{},{},{},{}
for u in range(u_len+1):
	user_movies[u] = filter(lambda x: R[u][x] > 0,movies)
	if(len(user_movies[u])):
		l = [R[u][i] for i in user_movies[u]]
	else:
		l = [0]
	mean_ratings[u] = np.mean(l)
	sigmas[u] = np.std(l)

for i in range(m_len+1):
	item_users[i] = filter(lambda x: R[x][i] > 0, users)
	if(len(item_users[i])):
		item_means[i] = np.mean([R[u][i] for u in item_users[i]])
	else:
		item_means[i] = 0

print("Baseline Predictor........................")

mu =  np.mean(train.rating)

def predict_baseline(u,i,K):
	lu,li,b_u,b_i = len(user_movies[u]), len(item_users[i]),0,0
	if(lu > 0):
		b_u = (mean_ratings[u] - mu)*lu/(lu+K)
	if(li > 0):
		b_i = (item_means[i] - b_u - mu)*li/(li+K)
	return mu + b_u + b_i

# num = (np.linspace(1,101,100)).astype(int)
# scores = []
# for k in num:
# 	y_pred = [predict_baseline(i[0],i[1],k) for i in X_test]
# 	score = np.mean((y - y_pred)**2)
# 	scores.append(score)

# # K = 25 is the optimal value

# plt.plot(num,scores,'ro')
# plt.show()
print("K-Nearest Neighbours.................................................")

def similarity(u1,u2,p=0):
	
	if(not p):
		# Pearson correlation coefficient for two items
		common_movies =  np.intersect1d(user_movies[u1],user_movies[u2],assume_unique=True)
		if(not len(common_movies)):
			l1,l2 = [0],[0]
		else:
			l1 = [R[u1][i] - mean_ratings[u1] for i in common_movies]
			l2 = [R[u2][i] - mean_ratings[u2] for i in common_movies]
	
	else:
		# Adjusted cosine similarity for two items
		common_users = np.intersect1d(item_users[u1],item_users[u2],assume_unique=True)
		if(not len(common_users)):
			l1,l2 = [0],[0]
		else:
			l1 = [R[u][u1] - mean_ratings[u] for u in common_users]
			l2 = [R[u][u2] - mean_ratings[u] for u in common_users]
	
	r = np.dot(l1,l2)
	
	if(r):
		r /= np.sqrt( np.sum(np.square(l1)) * np.sum(np.square(l2)))
	
	return r
	

list = {}
list[0],list[1] = users, movies 
def getNeighbours(u,p=0):
	similarities = []
	for i in list[p]:
		if(i != u):
			similarities.append((similarity(u,i,p),i))	
	similarities = sorted(similarities,reverse = True)
	return similarities

neighbours , neighbours1 = {},{}
# for i in users:
# 	neighbours[i] = getNeighbours(i)
# 	print i," done"

def predict(u,m,k):
	p,a,b = mean_ratings[u],0,0
	if(not neighbours.has_key(u)):
		neighbours[u] = getNeighbours(u)
		# print "oh"
	for i in range(k):
		u1 = neighbours[u][i][1]
		if(R[u1][m] != 0):
			if(sigmas[u1] != 0):
				a += neighbours[u][i][0]*(R[u1][m] - mean_ratings[u1])/sigmas[u1]
			b += np.abs(neighbours[u][i][0])
	if(b != 0):
		p += sigmas[u]*a/b;
	return p

def predict2(u,m,k):
	p,a,b = predict_baseline(u,m,25),0,0
	if(not neighbours1.has_key(m)):
		neighbours1[m] = getNeighbours(m,1)
		# print "yeah"
	for i in range(k):
		m1 = neighbours1[m][i][1]
		if(R[u][m1] != 0):
			a += neighbours1[m][i][0]*(R[u][m1] - p)
			b += np.abs(neighbours1[m][i][0])
	if(b):
		p += a/b;
	return p

def predict_all(u,m):
	p,a,b = mean_ratings[u],0,0
	for u1 in range(u_len+1):
		if(R[u1][m]):
			x = similarity(u,u1)
			if(sigmas[u1] != 0):
				a += x*(R[u1][m] - mean_ratings[u1])/sigmas[u1]
			b += np.abs(x)
	if(b):
		p += sigmas[u]*a/b;
	return p


num_neighbours = (np.linspace(20,900,45)).astype(int)
scores_u,scores_i,y_pred = [],[],[]

for k in num_neighbours:
	y_pred = [predict(i[0],i[1],k) for i in X_test]
	scores_u.append(np.mean((y - y_pred)**2))
	# print k,scores_u[-1]

y_pred = [predict_all(v[0],v[1]) for v in X_test]
print np.mean((y - y_pred)**2)

# num_neighbours = (np.linspace(20,1600,40)).astype(int)

for k in num_neighbours:
	y_pred = [predict2(i[0],i[1],k) for i in X_test]
	scores_i.append(np.mean((y - y_pred)**2))
	# print k,scores_i[-1]

plt.xlabel('No. of nearest neighbours')
plt.ylabel('RMSE')
user_based, = plt.plot(num_neighbours,scores_u,'bs')
item_based, = plt.plot(num_neighbours,scores_i,'ro')
plt.legend([user_based, item_based], ['User Based', 'Item Based'])
plt.show()

# with open('neighbours.txt','w') as f:
# 	for i in users:
# 		for item in neighbours[i]:
#   			f.write("%s\n" % item)

# with open('neighbours1.txt','w') as f:
# 	for i in users:
# 		for item in neighbours1[i]:
#   			f.write("%s\n" % item)