from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score

# Read in the training and dev labels
training_dict = {}
with open('bless2011/data_lex_train.tsv', 'r') as f:
    for line in f:
        word1, word2, label = line.strip().split('\t')
        training_dict[(word1, word2)] = (label == 'True')

dev_dict = {}
with open('bless2011/data_lex_val.tsv', 'r') as f:
    for line in f:
        word1, word2, label = line.strip().split('\t')
        dev_dict[(word1, word2)] = (label == 'True')

# create a mapping from word pair to list of all minimum deppaths
pairs2paths = defaultdict(list)
with open('new_wikipedia_deppaths.txt', 'r') as f:
    for line in f:
        word1, word2, deppath = line.strip().split('\t')
        pairs2paths[(word1, word2)].append(deppath)

# from paper, we only consider word pairs that had at least 5 distinct deppaths
filtered_pairs2paths = {pair: paths for pair, paths in pairs2paths.items() 
                        if len(paths) >= 5}
filtered_pairs2paths = dict(pairs2paths.items())

# create dev and train features in the form of deppath -> count
train_features = []
y_train = []
for pair, paths in filtered_pairs2paths.items():
    if pair in training_dict:
        train_features.append(dict(Counter(paths).items()))
        y_train.append(training_dict[pair])

dev_features = []
y_dev = []
for pair, paths in filtered_pairs2paths.items():
    if pair in dev_dict:
        dev_features.append(dict(Counter(paths).items()))
        y_dev.append(dev_dict[pair])

# convert features using DictVectorizer
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train_features)
X_dev = vectorizer.transform(dev_features)

clf1 = LogisticRegression(penalty='l2') # 0.4
#clf = LinearSVC() # 0.35294117647058826
clf2 = Perceptron() # really good... 0.69
clf3 = MultinomialNB() # pretty good... 0.5714285714285715

clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard', weights=[2,4,3])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_dev)

print('train score {}'.format(f1_score(y_true=y_train, y_pred=clf.predict(X_train))))
print('val score {}'.format(f1_score(y_true=y_dev, y_pred=y_pred)))

test_set = []
with open('bless2011/data_lex_test.tsv', 'r') as f:
    for line in f:
        word1, word2 = line.strip().split('\t')
        test_set.append((word1, word2))

counts = defaultdict(int)

with open('diy.txt', 'w') as f:
    for x in test_set:
        pred = False
        if x in filtered_pairs2paths:
            feature_vector = vectorizer.transform(dict(Counter(filtered_pairs2paths[x]).items()))
            pred = clf.predict(feature_vector)[0]
            counts[pred] += 1

        f.write('{}\t{}\t{}\n'.format(x[0], x[1], pred))

print(counts)