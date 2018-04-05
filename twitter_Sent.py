import twitter
import re
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#i am just trying to understand git kraken 




api = twitter.Api(consumer_key='nVf4oDAdoZxcSikwX6TFSgVwI',
                 consumer_secret='PPqSvNIavY8cPHnrEVYE8c49onuJvRKZXapzH9HWynfM9BSo1W',
                 access_token_key='3310846290-rYL6GhLGKx322JeU9Ei9Zc277Lxaqbo0om97dME',
                 access_token_secret='Av7QaYZweF7015pv71u4UwfzPJ3qIqmhoV3GzjUxfgZN6')

print(api.VerifyCredentials())

def createTestData(search_string):
    try:
        tweets_fetched=api.GetSearch(search_string, count=100)
        print "Great! We fetched " + str(len(tweets_fetched)) + " tweets with the term " + search_string + "!!"
        return [{"text": status.text, "label": None} for status in tweets_fetched]
    except:
        print "Sorry there was an error!"
        return None


search_string = input("Hi there! What are we searching for today?")
testData = createTestData(search_string)

print (testData[0:9])


####downloading the training data Neik sanders limited tweet version

def createLimitedTrainingCorpus(corpusFile,tweetDataFile):
    import csv
    corpus=[]
    with open(corpusFile,'rb') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2],"label":row[1],"topic":row[0]})

    trainingData = []
    for label in ["positive", "negative"]:
        i = 1
        for tweet in corpus:
            if tweet["label"] == label and i <= 50:
                try:
                    status = api.GetStatus(tweet["tweet_id"])
                    # Returns a twitter.Status object
                    print "Tweet fetched" + status.text
                    tweet["text"] = status.text
                    # tweet is a dictionary which already has tweet_id and label (positive/negative/neutral)
                    # Add another attribute now, the tweet text
                    trainingData.append(tweet)
                    i = i + 1
                except Exception, e:
                    print e

    with open(tweetDataFile, 'wb') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=',', quotechar="\"")
        # We'll add a try catch block here so that we still get the training data even if the write
        # fails
        for tweet in trainingData:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception, e:
                print e
    return trainingData


corpusFile="/Users/ravs/Downloads/sanders-twitter-0.2/sanders-twitter-0.2/corpus.csv"
tweetDataFile="/Users/ravs/Downloads/sanders-twitter-0.2/sanders-twitter-0.2/tweetDataFile.csv"



trainingData=createLimitedTrainingCorpus(corpusFile,tweetDataFile)




# A class to preprocess all the tweets, both test and training
# We will use regular expressions and NLTK for preprocessing
class PreProcessTweets:

    def __init__(self):

        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])


    def processTweets(self, list_of_tweets):
        # The list of tweets is a list of dictionaries which should have the keys, "text" and "label"
        processedTweets = []
        # This list will be a list of tuples. Each tuple is a tweet which is a list of words and its label
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]), tweet["label"]))
        return processedTweets

    def _processTweet(self, tweet):
            # 1. Convert to lower case
            tweet = tweet.lower()
            # 2. Replace links with the word URL
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
            # 3. Replace @username with "AT_USER"
            tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
            # 4. Replace #word with word
            tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
            # You can do further cleanup as well if you like, replace
            # repetitions of characters, for ex: change "huuuuungry" to "hungry"
            # We'll leave that as an exercise for you on regular expressions
            tweet = word_tokenize(tweet)
            # This tokenizes the tweet into a list of words
            # Let's now return this list minus any stopwords
            return [word for word in tweet if word not in self._stopwords]


tweetProcessor = PreProcessTweets()
ppTrainingData = tweetProcessor.processTweets(trainingData)
ppTestData = tweetProcessor.processTweets(testData)


print (ppTrainingData[:5])


# Naive Bayes Classifier - We'll use NLTK's built in classifier to perform the classification

# First build a vocabulary
def buildVocabulary(ppTrainingData):
    all_words=[]
    for (words,sentiment) in ppTrainingData:
        all_words.extend(words)
    # This will give us a list in which all the words in all the tweets are present
    # These have to be de-duped. Each word occurs in this list as many times as it
    # appears in the corpus
    wordlist=nltk.FreqDist(all_words)
    # This will create a dictionary with each word and its frequency
    word_features=wordlist.keys()
    # This will return the unique list of words in the corpus
    return word_features



# NLTK has an apply_features function that takes in a user-defined function to extract features
# from training data. We want to define our extract features function to take each tweet in
# The training data and represent it with the presence or absence of a word in the vocabulary

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
        # This will give us a dictionary , with keys like 'contains word1' and 'contains word2'
        # and values as True or False
    return features


# Now we can extract the features and train the classifier
word_features = buildVocabulary(ppTrainingData)
trainingFeatures=nltk.classify.apply_features(extract_features,ppTrainingData)
# apply_features will take the extract_features function we defined above, and apply it it
# each element of ppTrainingData. It automatically identifies that each of those elements
# is actually a tuple , so it takes the first element of the tuple to be the text and
# second element to be the label, and applies the function only on the text


NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
# We now have a classifier that has been trained using Naive Bayes



# Support Vector Machines

# We have to change the form of the data slightly. SKLearn has a CountVectorizer object.
# It will take in documents and directly return a term-document matrix with the frequencies of
# a word in the document. It builds the vocabulary by itself. We will give the trainingData
# and the labels separately to the SVM classifier and not as tuples.
# Another thing to take care of, if you built the Naive Bayes for more than 2 classes,
# SVM can only classify into 2 classes - it is a binary classifier.

svmTrainingData=[' '.join(tweet[0]) for tweet in ppTrainingData]
# Creates sentences out of the lists of words



vectorizer=CountVectorizer(min_df=1)
X=vectorizer.fit_transform(svmTrainingData).toarray()
# We now have a term document matrix

vocabulary = vectorizer.get_feature_names()

# Now for the twist we are adding to SVM. We'll use sentiwordnet to add some weights to these
# features

swn_weights=[]

for word in vocabulary:
    try:
        # Put this code in a try block as all the words may not be there in sentiwordnet (esp. Proper
        # nouns). Look for the synsets of that word in sentiwordnet
        synset = list(swn.senti_synsets(word))
        # use the first synset only to compute the score, as this represents the most common
        # usage of that word
        common_meaning =synset[0]
        # If the pos_Score is greater, use that as the weight, if neg_score is greater, use -neg_score
        # as the weight
        if common_meaning.pos_score()>common_meaning.neg_score():
            weight = common_meaning.pos_score()
        elif common_meaning.pos_score()<common_meaning.neg_score():
            weight =- common_meaning.neg_score()
        else:
            weight=0
    except:
        weight=0
    swn_weights.append(weight)


 #Let's now multiply each array in our original matrix with these weights
# Initialize a list

swn_X=[]
for row in X:
    swn_X.append(np.multiply(row,np.array(swn_weights)))
# Convert the list to a numpy array
swn_X=np.vstack(swn_X)



# We have our documents ready. Let's get the labels ready too.
# Lets map positive to 1 and negative to 2 so that everything is nicely represented as arrays
labels_to_array={"positive":1,"negative":2}
labels=[labels_to_array[tweet[1]] for tweet in ppTrainingData]
y=np.array(labels)


 #Let's now build our SVM classifier
from sklearn.svm import SVC
SVMClassifier=SVC()
SVMClassifier.fit(swn_X,y)




# First Naive Bayes
NBResultLabels=[NBayesClassifier.classify(extract_features(tweet[0])) for tweet in ppTestData]

# Now SVM
SVMResultLabels=[]
for tweet in ppTestData:
    tweet_sentence=' '.join(tweet[0])
    svmFeatures=np.multiply(vectorizer.transform([tweet_sentence]).toarray(),np.array(swn_weights))
    SVMResultLabels.append(SVMClassifier.predict(svmFeatures)[0])
    # predict() returns  a list of numpy arrays, get the first element of the first array
    # there is only 1 element and array


# Step 3 : GEt the majority vote and print the sentiment

if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print "NB Result Positive Sentiment" + str(100 * NBResultLabels.count('positive') / len(NBResultLabels)) + "%"
else:
    print "NB Result Negative Sentiment" + str(100 * NBResultLabels.count('negative') / len(NBResultLabels)) + "%"

if SVMResultLabels.count(1) > SVMResultLabels.count(2):
    print "SVM Result Positive Sentiment" + str(100 * SVMResultLabels.count(1) / len(SVMResultLabels)) + "%"
else:
    print "SVM Result Negative Sentiment" + str(100 * SVMResultLabels.count(2) / len(SVMResultLabels)) + "%"


print (testData[0:10])

print (NBResultLabels[0:10])

print (SVMResultLabels[0:10])





