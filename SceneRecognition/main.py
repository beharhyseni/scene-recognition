#Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
#based on a MATLAB code by James Hays and Sam Birch 

import numpy as np
from util import sample_images, build_vocabulary, get_bags_of_sifts
from classifiers import nearest_neighbor_classify, svm_classify
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

VOCAB_SIZE = 50
#For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

#For simplicity you can define a "num_train_per_cat" vairable, limiting the number of
#examples per category. num_train_per_cat = 100 for intance.

#Sample images from the training/testing dataset. 
#You can limit number of samples by using the n_sample parameter.

print('Getting paths and labels for all train and test data\n')
train_image_paths, train_labels = sample_images("C:/Users/behar/OneDrive/SceneRecognition/SceneRecognition/sift/train", n_sample=1000)
test_image_paths, test_labels = sample_images("C:/Users/behar/OneDrive/SceneRecognition/SceneRecognition/sift/test", n_sample=250)




''' Step 1: Represent each image with the appropriate feature
 Each function to construct features should return an N x d matrix, where
 N is the number of paths passed to the function and d is the 
 dimensionality of each image representation. See the starter code for
 each function for more details. '''

        
print('Extracting SIFT features\n')
#TODO: You code build_vocabulary function in util.py
kmeans = build_vocabulary(train_image_paths, vocab_size = VOCAB_SIZE)


#TODO: You code get_bags_of_sifts function in util.py 
train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)
test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)
#         
#If you want to avoid recomputing the features while debugging the
#classifiers, you can either 'save' and 'load' the extracted features
#to/from a file.'


# ***** IMPLEMENTATION OF THE AVERAGE HISTOGRAM *****
# count = 0
# all_images_indexes = []
# images_paths_names = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'InsideCity', 'Kitchen',
# 'LivingRoom', 'Mountain', 'Office', 'OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding']
#    
#    
#    
# # Loop through every path name (out of 15) in images_paths_names to save the indexes of each of the 15
# # image paths into a list.
# 
# for name_idx in range(0, len(images_paths_names)):
#     
#     image_indexes = []
#     path = images_paths_names[name_idx]
#     
#     for image_idx in range(0, len(train_image_paths)):    
#         image = train_image_paths[image_idx]          
#         
#         if path in image:            
#             image_indexes.append(image_idx)
#         
#     all_images_indexes.append(image_indexes)
# 
#             
#             
# 
# # Compute appropriate indexes and images
#         
# images_matrix = []
# for paths_indexes in all_images_indexes:
#     one_image_index_list = []
#     
#     for image_path in paths_indexes:
#         one_image_index_list.append(train_image_feats[image_path])
#     
#     images_matrix.append(one_image_index_list)
#     
# 
# # Average Histogram
# 
# averaged_matrix = []
# for matrix in images_matrix:
#     # Computes the average of arrays of a image scene (do this for each image scene = 15 image scenes).
#     avg = [float(sum(l))/len(l) for l in zip(*matrix)]
#     averaged_matrix.append(avg)
# 
# 
# for img_idx in range(len(averaged_matrix)):
#     img = averaged_matrix[img_idx]
#     n_bins = np.arange(0, VOCAB_SIZE, 1)
#     
#     plt.close()
#     
#     plt.bar(n_bins, img)
#     plt.xlabel('Number of Keywords (Vocab Size)')
#     plt.ylabel('Normalized Frequency')
# 
#     plt.title('Average Histogram for Category: ' + images_paths_names[img_idx] )
#     
#     plt.title(images_paths_names[img_idx])
#     plt.savefig("C:/Users/behar/OneDrive/SceneRecognition/SceneRecognition/Histograms/" + images_paths_names[img_idx] + ".png")

# ***** THE END OF IMPLEMENTATION OF THE AVERAGE HISTOGRAM *****

''' Step 2: Classify each test image by training and using the appropriate classifier
 Each function to classify test features will return an N x l cell array,
 where N is the number of test cases and each entry is a string indicating
 the predicted one-hot vector for each test image. See the starter code for each function
 for more details. '''








print('Using nearest neighbor classifier to predict test set categories\n')

pred_labels_knn = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)


# Confusion Matrix
the_confusion_matrix = confusion_matrix(test_labels, pred_labels_knn)
diagonal_sum = np.trace(the_confusion_matrix)
total_sum = np.sum(the_confusion_matrix)
print float(diagonal_sum)
print float(total_sum)

print the_confusion_matrix
accuracy = float(diagonal_sum)/float(total_sum)
print "KNN Accuracy: ", accuracy




print('Using support vector machine to predict test set categories\n')
#TODO: YOU CODE svm_classify function from classifers.py
# pred_labels_svm = svm_classify(train_image_feats, train_labels, test_image_feats)



print('---Evaluation---\n')
# Step 3: Build a confusion matrix and score the recognition system for 
#         each of the classifiers.
# TODO: In this step you will be doing evaluation. 
# 1) Calculate the total accuracy of your model by counting number
#   of true positives and true negatives over all. 
# 2) Build a Confusion matrix and visualize it. 
#   You will need to convert the one-hot format labels back
#   to their category name format.


# Interpreting your performance with 100 training examples per category:
#  accuracy  =   0 -> Your code is broken (probably not the classifier's
#                     fault! A classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .10 -> Your performance is chance. Something is broken or
#                     you ran the starter code unchanged.
#  accuracy ~= .50 -> Rough performance with bag of SIFT and nearest
#                     neighbor classifier. Can reach .60 with K-NN and
#                     different distance metrics.
#  accuracy ~= .60 -> You've gotten things roughly correct with bag of
#                     SIFT and a linear SVM classifier.
#  accuracy >= .70 -> You've also tuned your parameters well. E.g. number
#                     of clusters, SVM regularization, number of patches
#                     sampled when building vocabulary, size and step for
#                     dense SIFT features.
#  accuracy >= .80 -> You've added in spatial information somehow or you've
#                     added additional, complementary image features. This
#                     represents state of the art in Lazebnik et al 2006.
#  accuracy >= .85 -> You've done extremely well. This is the state of the
#                     art in the 2010 SUN database paper from fusing many 
#                     features. Don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> You used modern deep features trained on much larger
#                     image databases.
#  accuracy >= .96 -> You can beat a human at this task. This isn't a
#                     realistic number. Some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.