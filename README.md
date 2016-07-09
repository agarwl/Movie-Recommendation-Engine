In this code, we have implemented various collaborative filtering algorithms as different classes(models).
We have not used scikit learn in our code but have implemented all the algorithms by reading up from research 
papers but way of implementation is similar to scikit learn by encapsulating them in a black box.

There is not anything to test these algorithms as our project included their empirical analysis and implementation
and diggging into the details of how they work.
Currently, we are using the dataset ua.base and ua.test for training and testing respectively.
Other datasets are present in the dataset folder.For information about the datasets, please refer to the README
in the datasets folder.

The main.py contains the following models:

1. Baseline Predictor
2. NearestNeighbour
3. SVD

All the methods contains a 'Train' method:
- train : This method takes in two numpy arrays X and y where X denotes the feature set and Y denotes the values corresponding to those features.

The baseline predictor contains a predict method which takes in a userId and movieId as parameters and predicts
rating of the user for that movie.

The initialization of this model requires no parameters.

NearestNeighbour models takes in a number K as parameter which dentoes the no of neighbours to be used by the model.
It has the following three methods for predicting movie ratings (each of them takes a user id and movie as parameter):
1.  ubased_predict : User Based CF
2 . mbased_predict : Item Based CF
3 . similarityall_predict : All User Based CF with much smaller time complexity

SVD has a predict_rating function too using the userId and movieId as parameters.It also has a cross_validation error function which finds out the RMSE error on 5-fold cross validtion set. 
