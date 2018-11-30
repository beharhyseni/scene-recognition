import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
 
 #Starter code prepared by Borna Ghotbi for computer vision
 #based on MATLAB code by James Hay

'''This function will predict the category for every test image by finding
the training image with most similar features. Instead of 1 nearest
neighbor, you can vote based on k nearest neighbors which will increase
performance (although you need to pick a reasonable value for k). '''

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string 
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        
    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector 
        indicating the predicted category for each test image.

    Usefull funtion:
    	
    	# You can use knn from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    '''
    # Let M be the size of test_image_feats (as suggested in the function description above)
    M = len(test_image_feats)
    
    # Create a new empty array of size M x 1 (to be used to save predicted labels returned by the KNN predict)
    predicted_labels = np.empty((M,1))
    
    # Set the number of neigbors in the KNN function (saved into variable 'model' for convenience)
    model = KNeighborsClassifier(n_neighbors = 5) 
    
    # Fit the KNN model using 'train_image_feats' as training data and 'train_labels' as target values
    model.fit(train_image_feats, train_labels)
    
    # Save the predicted class labels for 'test_image_feats' into the variable 'labels'
    labels = model.predict(test_image_feats)
    
    # Since the variable 'labels' is not shaped in the required shape (it is not M x 1), we put every element of 'labels' into 
    # the correct cell of the array 'predicted_label' (this array has correct shape and dimensions: M x 1). 
    # We do this process by looping over each element in array 'labels' and saving each of these elements to the properly shaped array called 
    # 'predicted_labels'
    for label in range(len(labels)):
        predicted_labels[label] = labels[label]
    
    # return the predicted_labels which is a M x 1 array that contains the predicted class labels from the fitted KNN model
    return predicted_labels

'''This function will train a linear SVM for every category (i.e. one vs all)
and then use the learned linear classifiers to predict the category of
very test image. Every test feature will be evaluated with all 15 SVMs
and the most confident SVM will "win". Confidence, or distance from the
margin, is W*X + B where '*' is the inner product or dot product and W and
B are the learned hyperplane parameters. '''

def svm_classify(train_image_feats, train_labels, test_image_feats):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string 
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        
    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector 
        indicating the predicted category for each test image.

    Usefull funtion:
    	
    	# You can use svm from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/svm.html

    '''
    
    # Let M be the size of test_image_feats (as suggested in the function description above)
    M = len(test_image_feats)
    
    # Create a new empty array of size M x 1 (to be used to save predicted labels returned by the KNN predict)
    predicted_labels = np.empty((M,1))
    
    # Compute the One vs All classifier by fitting on train_image_feats and train_labels, and by predicting the test_image_feats labels. Argument C 
    # controls  how strongly regularized the model is. Lastly, Save the predicted labels into the variable 'labels'.
    # The OneVsRestClassifier evaluates all 15 classifiers on each test case and the classifier which is most confidently positive "wins". Also, 
    # each of these classifier will be trained to recognize, say, 'forest' vs 'non-forest', 'kitchen' vs 'non-kitchen', etc. also known as binary, in this case: 1-vs-15 SVMs.
    # So, it will will train 15 binary, 1-vs-all SVMs (here 1-vs-15), and this training will help to predict the labels of test_image_feats.
    labels = OneVsRestClassifier(SVC(C = 850)).fit(train_image_feats, train_labels).predict(test_image_feats) 
    
    # Since the variable 'labels' is not shaped in the required shape (it is not M x 1), we put every element of 'labels' into 
    # the correct cell of the array 'predicted_label' (this array has correct shape and dimensions: M x 1). 
    # We do this process by looping over each element in array 'labels' and saving each of these elements to the properly shaped array called 
    # 'predicted_labels'
    for label in range(len(labels)):
        predicted_labels[label] = labels[label]
    
    
    # return the predicted_labels which is a M x 1 array that contains the predicted class labels from the fitted SVM model
    return labels
    

