import numpy as np
import os
import glob
from sklearn.cluster import KMeans



def build_vocabulary(image_paths, vocab_size):
    """ Sample SIFT descriptors, cluster them using k-means, and return the fitted k-means model.
    NOTE: We don't necessarily need to use the entire training dataset. You can use the function
    sample_images() to sample a subset of images, and pass them into this function.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    vocab_size: the number of clusters desired.
    
    Returns
    -------
    kmeans: the fitted k-means clustering model.
    """
    n_image = len(image_paths)

    # Since want to sample tens of thousands of SIFT descriptors from different images, we
    # calculate the number of SIFT descriptors we need to sample from each image.
    n_each = int(np.ceil(10000 / n_image))  # You can adjust 10000 if more is desired
 
    # Initialize an array of features, which will store the sampled descriptors
    features = np.zeros((n_image * n_each, 128))
    
    # Initialize the int variable 'track_index' to 0. This variable will keep track of already filled rows in the 'features' array
    # and prevent replacing same feature's rows and columns that were already modified. (will see in more details how this work in the bottom)
    track_index = 0
    
    
    # Iterate through each image_paths. 'i' is the index of the current iteration's index while 'path' is the actual string of the current iteration.
    for i, path in enumerate(image_paths):
        
        # Load SIFT features from path
        descriptors = np.loadtxt(path, delimiter=',',dtype=float) 
        
        # Generate a random array of indexes taken from 'descriptors' array, where n_each is the size of this generated random array, 
        # len(descriptors) is the pool of integers that are generated and added into the array 'random_array_indexes". The argument 'len(descriptors)'
        # acts as being 'np.arange(len(descriptors)), which is a way to create an array, in this case integer elements, with minimum being 0 and max being the 'len(descriptors)'. 
        # This will act as the pool of integers to choose from. And lastly, the argument 'replace = True', means that when an image's descriptors length is less than n_each, 
        # then random integers can be same as some other random integers generated before. This will assure that when we save random descriptors into the 'features', there will be
        # no rows with all zero values. But, there may be some with repeated values, which is fine (according to the Professor's answer in Piazza post @333)
        random_array_indexes = np.random.choice(len(descriptors), n_each, replace = True)
        
        # Save random descriptors (a random descriptor is selected by using the current iteration's random_value as an index for descriptors). 
        # After we have selected the random descriptor (descriptors[random_value]), we save descriptor into the right place of 'features' by using the addition of track_index (initally 0) and counter 
        # as a value to move the "pointer" of the features into the next row. After one complete loop of random_array_indexes, we will have saved n_each rows into features. The next random_array_indexes
        # will be saved starting from features[track_index] which prevents same rows being replaced over and over again.
        for counter, random_value in enumerate(random_array_indexes):
            features[(track_index+counter)] = descriptors[random_value]
        
        # Update track_index after each completed loop by adding n_each to it. This will ensure the next loop will point to the correct (that is, not visited previously) 'features' row.
        track_index += n_each           
           
    # Compute KMeans clustering on 'features' with the number of clusters being the vocab_size 
    kmeans = KMeans(n_clusters = vocab_size).fit(features)
    
    return kmeans
    
def get_bags_of_sifts(image_paths, kmeans):
    """ Represent each image as bags of SIFT features histogram.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    kmeans: k-means clustering model with vocab_size centroids.

    Returns
    -------
    image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    """
    n_image = len(image_paths)
    vocab_size = kmeans.cluster_centers_.shape[0]
    image_feats = np.zeros((n_image, vocab_size))
        
    for i, path in enumerate(image_paths):
        
        # Load SIFT descriptors
        descriptors = np.loadtxt(path, delimiter=',',dtype=float)
           
        # Here we loop through every descriptor of the image_paths iteration in order to map each descriptor to the closest cluster center
        for descr_idx in range(0, len(descriptors)):
            descriptor = descriptors[descr_idx]
            
            # Predict the closest cluster center each sample in descriptor belongs to.
            # kmeans.predict function requires the argument to be reshaped in order to match the cluster center array.
            # descriptor.reshape(-1,len(descriptor)) will reshape the current descriptor into a len(descriptor) dimensional (ie, 128 D) array to make sure it
            # is correctly computed into kmeans.predict. 
            mapped_cluster = kmeans.predict(descriptor.reshape(-1,len(descriptor)))[0]
            
            # use 'i' as row index for image_feats and 'mapped_cluster' as column index in order to increment the correct bin for the predicted closes cluster center of the current descriptor
            image_feats[i][mapped_cluster] += 1
            
            
            
        # NORMALIZATION PART
        
        # Compute the sum of the current row in imge_feats and save this value to the variable 'the_sum'
        the_sum = np.sum(image_feats[i])
        
        # Iterate through each column in the current row ('i') of image_feats
        for column_index in range(0, len(image_feats[i])):
            
            # Divide each column value in the current image_feats row with the_sum (total sum of all columns in the current row) and save this value into variable 'normalized_value"
            normalized_value = np.divide(image_feats[i][column_index], the_sum)
            
            # Replace each column in of the current image_feats row with the calcualted normalized value.
            # This will ensure that every image_feats row will sum up to one; Hence, it is now normalized.
            image_feats[i][column_index] = normalized_value

    # Return the normalized image_feats
    return image_feats

def sample_images(ds_path, n_sample):
    """ Sample images from the training/testing dataset.

    Parameters
    ----------
    ds_path: path to the training/testing dataset.
             e.g., sift/train or sift/test
    n_sample: the number of images you want to sample from the dataset.
              if None, use the entire dataset. 
    
    Returns
    -------
    image_paths: a (n_sample, 1) array that contains the paths to the descriptors. 
    """
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.txt"))
    n_files = len(files)

    if n_sample == None:
        n_sample = n_files

    # Randomly sample from the training/testing dataset
    # Depending on the purpose, we might not need to use the entire dataset
    idx = np.random.choice(n_files, size=n_sample, replace=False)
    image_paths = np.asarray(files)[idx]
 
    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_sample)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0,0]

    return image_paths, labels

