import pandas as pd
import numpy as np

class Wrapper(object):

    def __init__(self):
        self.user_movies = {}
        self.item_users = {}
        self.item_means = {}
        self.mean_ratings = {}
        self.sigmas = {}
        self.R = 0

        self.users = 0
        self.movies = 0
        self.mu = 0
        self.u_len = 0
        self.m_len = 0

    def preprocess(self, X, Y):

        # to store the unique ids of users
        self.users = np.unique(X[:, 0])
        # to store the unique ids of movies 
        self.movies = np.unique(X[:, 1])
        self.u_len = X[:, 0].max()
        self.m_len = X[:, 1].max()

        # R is the rating matrix where R[u][m] gives the rating for movie m by user u 
        self.R = np.zeros(shape=(max(self.users) + 1, max(self.movies) + 1))

        for i, x in enumerate(X):
            self.R[x[0]][x[1]] = Y[i]

        # Preprocessing the data
        for u in range(self.u_len + 1):
            self.user_movies[u] = filter(lambda x: self.R[u][x] > 0, self.movies)
            if(len(self.user_movies[u])):
                l = [self.R[u][i] for i in self.user_movies[u]]
            else:
                l = [0]
            self.mean_ratings[u] = np.mean(l)
            self.sigmas[u] = np.std(l)

        for i in range(self.m_len + 1):
            self.item_users[i] = filter(lambda x: self.R[x][i] > 0, self.users)
            if(len(self.item_users[i])):
                self.item_means[i] = np.mean([self.R[u][i] for u in self.item_users[i]])
            else:
                self.item_means[i] = 0

        self.mu =  np.mean(Y)

class BaseLine(Wrapper):

    def __init__(self, K_baseline=25):
        super(BaseLine, self).__init__()
        self.K_baseline = K_baseline

    def train(self, X, Y):
        Wrapper.preprocess(self, X, Y)

    def predict(self, u, i):

        lu, li, b_u, b_i = len(self.user_movies[u]), len(self.item_users[i]), 0, 0
        if(lu > 0):
            b_u = (self.mean_ratings[u] - self.mu)*lu/(lu+self.K_baseline)
        if(li > 0):
            b_i = (self.item_means[i] - b_u - self.mu)*li/(li+self.K_baseline)
        return self.mu + b_u + b_i


# Nearest Neighbour Model
# Parameters : No of Neighbours to be used for prediction
class NearestNeighbours(BaseLine):

    def __init__(self, neigbours):
        super(NearestNeighbours, self).__init__()
        # no of neighbours required to predict
        self.neigbour_count = neigbours
        # to store the users closest to a particular user using the Pearson coefficient of similarity
        self.neighbours_user = {}
        # to store the movies closest to a particular movie using the Cosine Similarity
        self.neighbours_movie = {}

    def train(self, X, Y):
        Wrapper.preprocess(self, X, Y)

    def similarity(self, u1, u2, method):
        if(method == "user"):
            # Pearson correlation coefficient for two users
            common_movies =  np.intersect1d(self.user_movies[u1],self.user_movies[u2],assume_unique=True)
            if(not len(common_movies)):
                l1,l2 = [0],[0]
            else:
                l1 = [self.R[u1][i] - self.mean_ratings[u1] for i in common_movies]
                l2 = [self.R[u2][i] - self.mean_ratings[u2] for i in common_movies]
        elif(method =="movie"):
            # Adjusted cosine similarity for two items
            common_users = np.intersect1d(self.item_users[u1],self.item_users[u2],assume_unique=True)
            if(not len(common_users)):
                l1,l2 = [0],[0]
            else:
                l1 = [self.R[u][u1] - self.mean_ratings[u] for u in common_users]
                l2 = [self.R[u][u2] - self.mean_ratings[u] for u in common_users]
        
        r = np.dot(l1,l2)
        if(r):
            r /= np.sqrt( np.sum(np.square(l1)) * np.sum(np.square(l2)))
        
        return r

    # finding the neighbours by iterating over all items or user
    def getNeighbours(self, u, p):

        similarities = []
        if p == "user":
            for i in self.users:
                if(i != u):
                    similarities.append((self.similarity(u,i,p),i))
        elif p == "movie":
            for i in self.movies:
                if(i != u):
                    similarities.append((self.similarity(u,i,p),i))  
        similarities = sorted(similarities, reverse=True)
        return similarities

    # KNN based on User based CF
    def ubased_predict(self, u,m):
        p,a,b = self.mean_ratings[u],0,0
        if(not self.neighbours_user.has_key(u)):
            self.neighbours_user[u] = self.getNeighbours(u, "user")
            # print "oh"
        for i in range(self.neigbour_count):
            u1 = self.neighbours_user[u][i][1]
            if(self.R[u1][m] != 0):
                if(sigmas[u1] != 0):
                    a += self.neighbours_user[u][i][0]*(self.R[u1][m] - self.mean_ratings[u1])/sigmas[u1]
                b += np.abs(self.neighbours_user[u][i][0])
        if(b != 0):
            p += sigmas[u]*a/b;
        return p

    #KNN based on Item Based CF
    def mbased_predict(self, u,m):
        p,a,b = self.predict(u, m), 0, 0
        if(not self.neighbours_movie.has_key(m)):
            self.neighbours_movie[m] = self.getNeighbours(m, "movie")
            # print "yeah"
        for i in range(self.neigbour_count):
            m1 = self.neighbours_movie[m][i][1]
            if(self.R[u][m1] != 0):
                a += self.neighbours_movie[m][i][0]*(self.R[u][m1] - p)
                b += np.abs(self.neighbours_movie[m][i][0])
        if(b):
            p += a/b;
        return p

    # A nearest nighbour model computing the rating using all those users who have rated the movie 
    # and weighting their ratings by clossness to the user
    def similarityall_predict(self, u,m):
        p,a,b = self.mean_ratings[u],0,0
        for u1 in range(self.u_len+1):
            if(self.R[u1][m]):
                x = self.similarity(u, u1, "user")
                if(self.sigmas[u1] != 0):
                    a += x*(self.R[u1][m] - self.mean_ratings[u1])/self.sigmas[u1]
                b += np.abs(x)
        if(b):
            p += self.sigmas[u]*a/b;
        return p

#Defined a class for for implementing incremental SVD algorithm
class SVD(Wrapper):
    # Initiailizing all the weights and and the dimension of features required along with 
    # the constants for the regularization and bias terms
    def __init__(self, epochs=100, dim=60, learning_rate=0.007, k_u=0.05, k_m=0.05,k_b=0.05,bias=False):
        self.dim = dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.k_u = k_u
        self.k_m = k_m
        self.k_b=k_b
        self.bias = bias
    # a function which predicts the rating for a given user and movie 
    def predict_rating(self, user, movie):
        x = np.dot(self.U[user], self.M[movie])
        if(self.bias == True):
            x += mu + self.alpha[user] + self.beta[movie]
        return x

    # splits the data into test and train for cross validation purposes (n fold splitting)
    def test_train_split(self, X, n):
        np.random.shuffle(X)
        l = (int)(X.shape[0] * (n - 1) / n)
        self.X_train, self.X_test = X[:l,:], X[l:,:]
        self.y_test = self.X_test[:, 2]

    # trains the parameters on the train data until validation error starts to increase
    def train(self, X,Y):
        Wrapper.preprocess(self, X, Y)
        self.U = np.empty((self.u_len + 1, self.dim))
        self.M = np.empty((self.m_len + 1, self.dim))
        self.alpha = np.zeros(self.u_len+1)
        self.beta = np.zeros(self.m_len+1)
        initial_val = np.random.uniform(-0.01,0.01)
        for row in self.U:
            row.fill(initial_val)
        for row in self.M:
            row.fill(initial_val)
        self.test_train_split(X,5)
        err = float('Inf')
        U_temp = np.zeros(self.dim)
        for r in range(self.epochs):
            # for all examples in the training set, update the feature vector for the user and
            # the movie corresponding to that user
            for x in self.X_train:
                u, m = x[0], x[1]
                error = R[u][m] - self.predict_rating(u, m)
                # Updates similar to the Stochastic gradient descent
                U_temp = self.U[u] + self.learning_rate * (error * self.M[m] - self.U[u] * self.k_u)
                self.M[m] += self.learning_rate * (error * self.U[u] - self.M[m] * self.k_m)
                self.U[u] = U_temp[:]
                # If biases are used, then update them too
                if(self.bias == True):
                    self.alpha[u] += self.learning_rate*(error - self.k_b*self.alpha[u])
                    self.beta[m] += self.learning_rate*(error - self.k_b*self.beta[m])
            curr_err = self.validation_error()
            if(curr_err < err):
                err = curr_err
                # print r, ":", err
            else:
                break

    def validation_error(self):
            return self.test(self.X_test,self.y_test)

    def test(self, X, Y):
        y_pred = [self.predict_rating(i[0], i[1]) for i in X]
        return np.mean((Y - y_pred)**2)


with open('datasets/ua.base', 'r') as f:
    train = pd.read_csv(f)
    train = train.drop('timestamp', 1)

X = train.values
Y = train['rating'].values
X = np.delete(X, 2, axis=1)

# Define a KNN model with parameter value equal to 20
KNN = NearestNeighbours(20)
KNN.train(X, Y)

with open('datasets/ua.test','r') as f:
    test = pd.read_csv(f)
    test = test.drop('timestamp',1)

X_test = test.values
y_test = test['rating'].values
X_test = np.delete(X_test,2,axis = 1)

print "Prediction for user 1 and movie 10 is", KNN.similarityall_predict(1, 10)

# For plotting related to KNN

# scores_i,y_pred = [],[]
# num_neighbours = (np.linspace(2,2,1)).astype(int)

# for k in num_neighbours:
#     y_pred = [KNN.mbased_predict(i[0],i[1]) for i in X_test]
#     scores_i.append(np.mean((y - y_pred)**2))
#     print k,scores_i[-1]

# plt.xlabel('No. of nearest neighbours')
# plt.ylabel('RMSE')
# item_based = plt.plot(num_neighbours,scores_i,'ro')
# plt.show()

# For plotting the plots related to SVD 

# learning_rates = np.linspace(0.001,0.1,25)
# alphas = np.logspace(-4,4,9)
# dims = np.linspace(50,600,30).astype(int)
# error = float("Inf")
# # prevtime = time.clock()
# # for learning_rate in learning_rates:
# # for alpha in alphas:
# # for dim in dims:
#     # svd = SVD(dim=40,k_u=alpha,k_m=alpha,bias=True)
#     svd = SVD(k_b=alpha, bias=True)
#     svd.train(X)
#     curr = svd.validation_error()
#     if(curr < error):
#         error = curr
#     rmse = svd.test(X_test,y)
#     # curtime = time.clock()
#     print alpha, rmse
#     # prevtime = curtime