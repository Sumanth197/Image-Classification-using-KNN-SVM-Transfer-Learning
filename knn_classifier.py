import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

Visual_bag=[]


# K-Means Clustering
for i in range(1, 1889):
    training_file_sift_name = str(i) + '_train_sift.csv'
    training_file_sift_path = os.path.join("/home/sumanth/Desktop/hw2_data/train_sift_features/",training_file_sift_name)
    df = pd.read_csv(training_file_sift_path, header=None)
    df.drop(df.index[:4], axis=1, inplace=True)
    arr = np.asarray(df)
    #print (len(arr)) 
    for j in range(len(arr)):
        Visual_bag.append(arr[j][:])

Visual_bag = np.asarray(Visual_bag)
Kmeans = KMeans(n_clusters=128,random_state = 0).fit(Visual_bag)


print ("Clustering Done\n")



# Visual Bag for Training Set 

total_training_words=[]
for i in range(1, 1889):
    training_file_name = str(i) + '_train_sift.csv'
    training_file_path = os.path.join("/home/sumanth/Desktop/hw2_data/train_sift_features/",training_file_name)
    df = pd.read_csv(training_file_path, header=None)
    df.drop(df.index[:4], axis=1, inplace=True)
    arr = np.asarray(df)
    #print (len(arr))
    file_training_word = []
    for j in range(len(arr)):
    	file_training_word.append(arr[j][:])	
    total_training_words.append(Kmeans.predict(file_training_word))

total_training_words = np.array(total_training_words)


print ("Visual Bag has Training features\n")




# Train HISTOGRAM


training_bag = []
for i in range(1888):
	training_bag_words = np.zeros(128)
	for j in range(len(total_training_words[i])):
		training_bag_words[ total_training_words[i][j] ] += 1
	training_bag.append(training_bag_words)

training_bag = np.asarray(training_bag).astype(int)



training_labels = pd.read_csv('/home/sumanth/Desktop/hw2_data/train_labels.csv')
training_labels = training_labels.columns
training_labels = np.asarray(training_labels).astype('float32')
training_labels = training_labels.astype(int)

print ("Training Labels done\n")




#Visual Bag  for Testing Set

total_testing_words=[]
for i in range(1, 801):
    test_sift_file_name = str(i) + '_test_sift.csv'
    test_sift_file_path = os.path.join("/home/sumanth/Desktop/hw2_data/test_sift_features/",test_sift_file_name)
    df = pd.read_csv(test_sift_file_path, header=None)
    df.drop(df.index[:4], axis=1, inplace=True)
    arr = np.asarray(df)
    testing_word = []
    for j in range(len(arr)):
    	testing_word.append(arr[j][:])
    total_testing_words.append(Kmeans.predict(testing_word))
total_testing_words = np.asarray(total_testing_words)


print("Visual Bag of Testing has done\n")




# TEST HISTOGRAM

testing_bag = []

for i in range(800):
    test_bag_words = np.zeros(128)
    for j in range(len(total_testing_words[i])):
	    test_bag_words[total_testing_words[i][j]] += 1
    testing_bag.append(test_bag_words)
	
testing_bag = np.asarray(testing_bag).astype(int) 

 
testing_labels = pd.read_csv('/home/sumanth/Desktop/hw2_data/test_labels.csv')
testing_labels = testing_labels.columns
testing_labels = np.asarray(testing_labels).astype('float32')
testing_labels = testing_labels.astype(int)

print ("Testing Labels done\n")

# KNN CLASSIFICATION
#print '#########################KNN############################'
neighbour = KNeighborsClassifier(n_neighbors=16)
neighbour.fit(training_bag, training_labels)
train_accuracy=neighbour.score(training_bag, training_labels)
print(train_accuracy)
#print '######################TESTING###########################'

test_predictions = np.zeros(800)
accuracy_count = 0
for i in range(800):
    p = neighbour.predict( [ testing_bag[i] ] )
    test_predictions[i] = p
    if (p - testing_labels[i]) == 0:
        accuracy_count += 1

print (accuracy_count)

accuracy = float(accuracy_count)/800
accuracy = accuracy * 100

test_predictions = np.asarray(test_predictions).astype(int)

#print test_predictions
print("\n")
print (len(test_predictions))
print("\n")
print ('accuracy:' + str	(accuracy))



#Finding Confusion Matrix

cnf_matrix = confusion_matrix(testing_labels, test_predictions)

print(cnf_matrix)

'''confusion_matrix = np.zeros(shape=(8,8))

for i in range(len(test_predictions)):
	confusion_matrix[testing_labels[i]][test_predictions[i]] +=1

print (confusion_matrix)'''
