import gensim, logging
import numpy as np  # Make sure that numpy is imported
import math
import random
import sys
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


'''
This file takes in the output of the runTagger bash script and outputs a file of feature data
for every tweet from a CrowdFlower job. The next script to be run, on the output
of this one, is "arrayify_twokenizer_output_feature_data.py"
'''

def process_txt_data() :
	# ./runTagger.sh tweets2.txt > parsed_tweets.txt
	# valence.txt and arousal.txt are files of the CrowdFlower-labelled scores 
	# (averaged over CrowdFlower participants) in the same order as the other features.
	# political_tweets.txt is a file of just the content of the tweets from CrowdFlower.
	# parsed_tweets.txt would be a file of tokenized and POS labelled tweets, created by the bash script shown above.
	# right now, this function takes out all sarcastic tweets
	print("Processing raw .txt file data...")
	f = open('feature_labelled_data.txt', 'w')
	a = open('arousal_labels.txt', 'r')
	v = open('valence_labels.txt', 'r')
	t = open('political_tweets.txt', 'r') 
	s = open('sarcastic_labels.txt', 'r')
	c = open('numeric_class_by_eighth.txt', 'r')
	tweets = t.readlines()
	valences = v.readlines()
	arousals = a.readlines()
	sarcasm = s.readlines()
	classes = c.readlines()
	print "Length of data sets we are reading:"
	print len(valences)
	print len(arousals)
	print len(tweets)
	print len(sarcasm)
	print len(classes)
	corpus = []
	valence_set = []
	arousal_set = []
	sarcastic_set = []
	classes_set = []
	counter = 0
	for i in range(len(tweets)):
		tweet = tweets[i].strip()
		counter += 1
		valence = valences[i].strip()
		arousal = arousals[i].strip()
		sarcastic = sarcasm[i].strip()
		c = classes[i].strip()
		tweet = tweet.split(' ')
		if sarcastic != 1:
			corpus.append(tweet)
			f.write(str(tweet))
			valence_set.append(valence)
			arousal_set.append(arousal)
			classes_set.append(c)
			# tokens = line[0]
			# POS = line[1]
			# scores = line[2]
			# content = line[3]
			# f.write(tokens + '\t' + POS + '\t' + label)
			f.write('\n')

	# d.close()
	t.close()
	# y.close()
	f.close()
	s.close()
	a.close()
	v.close()
	return corpus, valence_set, arousal_set, classes_set


def separate_train_and_test(corpus, arousal, valence, classes):
	# Take out test set
	print ("Separating test and training data...")
	training_data = []
	testing_data = []
	arousal_training_data = []
	arousal_testing_data = []
	valence_training_data = []
	valence_testing_data = []
	class_training_data = []
	class_testing_data = []
	print classes
	ten_percent = int(0.1*len(corpus))
	testing_indices = random.sample(xrange(len(corpus)), ten_percent)
	for i in range(len(corpus)):
		if i in testing_indices:
			testing_data.append(corpus[i])
			arousal_testing_data.append(arousal[i])
			valence_testing_data.append(valence[i])
			class_testing_data.append(classes[i])
		else:
			training_data.append(corpus[i])
			arousal_training_data.append(arousal[i])
			valence_training_data.append(valence[i])
			class_training_data.append(classes[i])
	return testing_data, training_data, arousal_training_data, arousal_testing_data, valence_training_data, valence_testing_data, class_training_data, class_testing_data


def train_word2vec(training_data, num_features):
		# Train Word2Vec here!
		print("Training word2vec...")
		# This model runs two passes: the first creates a dictionary of all words in the corpus that appear > 1 time, 
		# the second trains the neural net with 200 features, in 4 parallel threads.
		model = gensim.models.Word2Vec(corpus, min_count = 2, size = num_features, workers=4)
		# Persist the model to disk
		model.save('/tmp/political_word2vec_model')
		# The next line saves RAM once you're done training
		model.init_sims(replace=True)
		return model

def makeFeatureVec(words, model, num_features, lowest, highest):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # Gather info for normalizing data (comment out if not doing multionimalNB)
    for i in featureVec:
		if i < lowest:
			lowest = i
		if i > highest: 
			highest = i
    # Divide the result by the number of words to get the average
    if nwords != 0.0:
	    featureVec = np.divide(featureVec,nwords)
    return featureVec, lowest, highest


def getAvgFeatureVecs(tweets, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    tweetFeatureVecs = np.zeros((len(tweets),num_features),dtype="float32")
    # 
    lowest = sys.maxint
    highest = 0.0
    # Loop through the reviews
    for tweet in tweets:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Tweet %d of %d" % (counter, len(tweets))
       # 
       # Call the function (defined above) that makes average feature vectors
       tweetFeatureVecs[counter], lowest, highest = makeFeatureVec(tweet, model, \
           num_features, lowest, highest)
       #
       # Increment the counter
       counter = counter + 1.
    for t in range(len(tweetFeatureVecs)):
    	for w in range(len(tweetFeatureVecs[t])):
    		tweetFeatureVecs[t][w] = (float(tweetFeatureVecs[t][w]) - lowest)/(highest - lowest)
    return tweetFeatureVecs




if __name__ == "__main__": 
	# Clean and separate data:
	corpus, valence, arousal, classes = process_txt_data()
	test_data, train_data, arousal_training_data, arousal_testing_data, valence_training_data, valence_testing_data, class_training_data, class_testing_data = separate_train_and_test(corpus, arousal, valence, classes)
	f = open('labels.csv', 'wb')
	csv_writer = csv.writer(f)
	# Train word2vec + decision tree regression:
	# This is the number of features associated with each word in our word2vec transformation
	num_features = 50
	model = train_word2vec(train_data, num_features)
	print "Creating average feature vecs for trainint tweets"
	trainDataVecs = getAvgFeatureVecs( train_data, model, num_features )
	print "Creating average feature vecs for test tweets"
	testDataVecs = getAvgFeatureVecs( test_data, model, num_features )
	print testDataVecs[0]
	# Fit a decision tree regression to the training data, with depth = depth
	depth = 10
	tree1 = DecisionTreeRegressor(max_depth = depth)
	tree2 = DecisionTreeRegressor(max_depth = depth)

	print "Fitting a decision tree to labeled training data and valence..."
	valence_tree = tree1.fit( trainDataVecs,  valence_training_data)
	filename = 'valence_tree.sav'
	pickle.dump(valence_tree, open(filename, 'wb'))

	print "Fitting a decision tree to labeled training data and arousal..."
	arousal_tree = tree2.fit( trainDataVecs, arousal_training_data)
	filename = 'arousal_tree.sav'
	pickle.dump(arousal_tree, open(filename, 'wb'))
	# Test & extract results 
	print "Running trained regression on test data..."
	valence_result = valence_tree.predict( testDataVecs )
	arousal_result = arousal_tree.predict( testDataVecs )

	valence_in_sample = valence_tree.predict(trainDataVecs)
	arousal_in_sample = arousal_tree.predict(trainDataVecs)

	# out_of_sample_0_1_error_counter_arousal = 0.0
	# for i in range(len(testDataVecs)):
	# 	print abs(float(arousal_result[i]) - float(arousal_testing_data[i]))
	# 	if (abs(float(arousal_result[i]) - float(arousal_testing_data[i])) > 25):
	# 		out_of_sample_0_1_error_counter_arousal += 1.0
	# out_of_sample_0_1_error_arousal = out_of_sample_0_1_error_counter_arousal/len(testDataVecs)
	# print "Out of sample error arousal: "
	# print out_of_sample_0_1_error_arousal

	# out_of_sample_0_1_error_counter_valence = 0.0
	# for j in range(len(testDataVecs)):
	# 	if (abs(float(valence_result[j]) - float(valence_testing_data[j])) > 25):
	# 		# print abs(float(valence_result[j]) - float(valence_testing_data[j]))
	# 		# print valence_result[j]
	# 		# print valence_testing_data[j]
	# 		out_of_sample_0_1_error_counter_valence += 1.0
	# out_of_sample_0_1_error_valence = out_of_sample_0_1_error_counter_valence/len(testDataVecs)
	# print "Out of sample error valence: "
	# print out_of_sample_0_1_error_valence
	out_of_sample_error_cosine_similarity = 0.0
	in_sample_error_cosine_similarity = 0.0
	bad_result_points_x = []
	bad_result_points_y = []
	good_result_points_x = []
	good_result_points_y = []
	real_points_x = []
	real_points_y = []
	for i in range(len(testDataVecs)):
		test = []
		result = []
		test.append(valence_testing_data[i])
		test.append(arousal_testing_data[i])
		result.append(valence_result[i])
		result.append(arousal_result[i])
		csv_writer.writerow(result)
		cos = cosine_similarity(test, result)
		angle_in_radians = math.acos(cos)
		angle_in_degrees = math.degrees(angle_in_radians)

		if angle_in_degrees > 22.5:
			print result
			bad_result_points_x.append(result[0])
			bad_result_points_y.append(result[1])
			real_points_x.append(test[0])
			real_points_y.append(test[1])
			out_of_sample_error_cosine_similarity += 1
		else: 
			good_result_points_x.append(result[0])
			good_result_points_y.append(result[1])

	# Method of finding error when 4-class classification (above the arousal axis)
	# for i in range(len(testDataVecs)):
	# 	test = []
	# 	result = []
	# 	test.append(valence_testing_data[i])
	# 	test.append(arousal_testing_data[i])
	# 	result.append(valence_result[i])
	# 	result.append(arousal_result[i])
	# 	# arousal 0 is now at 100, valence 0 is still at 0
	# 	if (result[1] >= 75 and test[1] < 75):
	# 		bad_result_points_x.append(result[0])
	# 		bad_result_points_y.append(result[1])
	# 		real_points_x.append(test[0])
	# 		real_points_y.append(test[1])
	# 		out_of_sample_error_cosine_similarity += 1
	# 	elif (result[1] < 75 and test[1] >= 75):
	# 		bad_result_points_x.append(result[0])
	# 		bad_result_points_y.append(result[1])
	# 		real_points_x.append(test[0])
	# 		real_points_y.append(test[1])
	# 		out_of_sample_error_cosine_similarity += 1
	# 	else: 
	# 		good_result_points_x.append(result[0])
	# 		good_result_points_y.append(result[1])
	out_of_sample_error_cosine_similarity = out_of_sample_error_cosine_similarity / len(testDataVecs)
	print "out of sample error:"
	print out_of_sample_error_cosine_similarity

	# for i in range(len(trainDataVecs)):
	# 	train = []
	# 	result = []
	# 	train.append(valence_training_data[i])
	# 	train.append(arousal_training_data[i])
	# 	result.append(valence_in_sample[i])
	# 	result.append(arousal_in_sample[i])
	# 	cos = cosine_similarity(train, result)
	# 	print cos
	# 	angle_in_radians = math.acos(cos)
	# 	angle_in_degrees = math.degrees(angle_in_radians)

	# 	if angle_in_degrees > 45:
	# 		in_sample_error_cosine_similarity += 1
	# in_sample_error_cosine_similarity = in_sample_error_cosine_similarity / len(trainDataVecs)
	# print "in sample error:"
	# print in_sample_error_cosine_similarity

	plt.scatter(bad_result_points_x, bad_result_points_y, color='red')
	plt.scatter(real_points_x, real_points_y, color='green')
	plt.scatter(good_result_points_x, good_result_points_y, color = "blue")
	plt.plot((-150, 150), (0,0), color="black")
	plt.plot((0, 0), (-150, 150), color = "black")
	plt.ylabel("Arousal")
	plt.xlabel("Valence")
	plt.title("Emotional Circumplex- 22.5 Degree Agreement")
	agreement = mpatches.Patch(color='blue', label='Agreement Data Points')
	disagreement = mpatches.Patch(color='red', label='Disagreement Data Points - Results')
	truth = mpatches.Patch(color='green', label='Disagreement Data Points - Truth')
	plt.legend(handles=[agreement, disagreement, truth], loc = 'lower left')
	plt.show()
	# Write the test results 
	# output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	# output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

	# Train multinomial naive bayes based on word2vec averages over a tweet
	multinomialNB_model = MultinomialNB()
	multinomialNB_model.fit(trainDataVecs, class_training_data)
	# Predict held-out test data
	multinomialNB_result = multinomialNB_model.predict(testDataVecs)
	# Evaluate error: 0/1
	out_of_sample_error_multinomialNB = 0.
	for i in range(len(testDataVecs)):
		print multinomialNB_result[i]
		print class_testing_data[i]
		if multinomialNB_result[i] != class_testing_data[i]:
			out_of_sample_error_multinomialNB += 1.
		#else:
			#print multinomialNB_result[i]
	print "unnormalized error"
	print out_of_sample_error_multinomialNB
	out_of_sample_error_multinomialNB = out_of_sample_error_multinomialNB / float(len(testDataVecs))
	print "Out of sample MultinomialNB error:"
	print out_of_sample_error_multinomialNB
	f.close()

