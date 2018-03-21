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

from itertools import product
from gensim.models import KeyedVectors
vecfile = 'paragram_300_sl999.txt.filter'
vecs = KeyedVectors.load_word2vec_format(vecfile)


def get_closest_pair(all_pairs, missing_pair):
    if missing_pair[1] not in vecs or missing_pair[1] not in vecs:
        return

    w1s = [(missing_pair[0], 1)]
    w2s = [(missing_pair[1], 1)]
    w1s.extend(vecs.most_similar(missing_pair[0]))
    w2s.extend(vecs.most_similar(missing_pair[1]))

    new_pairs = []

    for (w1, s1), (w2, s2) in product(w1s, w2s):
        new_pair = (w1, w2)
        if new_pair in all_pairs:
            new_pairs.append((new_pair, s1 * s2))
    
    if len(new_pairs) > 0:
        best_pair = max(new_pairs, key=lambda x: x[1])
        return best_pair[0]


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
# we don't have enough deppaths so we don't do this
filtered_pairs2paths = dict(pairs2paths.items())

# create dev and train features in the form of deppath -> count
train_features = []
y_train = []
for pair, paths in filtered_pairs2paths.items():
    if pair in training_dict:
        train_features.append(dict(Counter(paths).items()))
        y_train.append(training_dict[pair])
    else:
        closest_pair = get_closest_pair(training_dict, pair)
        if closest_pair:
            train_features.append(dict(Counter(paths).items()))
            y_train.append(training_dict[closest_pair])

dev_features = []
y_dev = []
for pair, paths in filtered_pairs2paths.items():
    if pair in dev_dict:
        dev_features.append(dict(Counter(paths).items()))
        y_dev.append(dev_dict[pair])
    else:
        closest_pair = get_closest_pair(dev_dict, pair)
        if closest_pair:
            train_features.append(dict(Counter(paths).items()))
            y_train.append(dev_dict[closest_pair])

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

def print_stats(dataset, y_true, y_pred):
    print('{} precision {}'.format(dataset, precision_score(y_true, y_pred)))
    print('{} recall {}'.format(dataset, recall_score(y_true, y_pred)))
    print('{} f-score {}'.format(dataset, f1_score(y_true, y_pred)))

print_stats('train', y_train, clf.predict(X_train))
print_stats('val', y_dev, clf.predict(X_dev))

hearst = []
with open('pair_extracts.txt', 'r') as f:
    for line in f:
        w1, w2 = line.strip().split('\t')
        hearst.append((w1, w2))

counts = defaultdict(int)

test_set = []
with open('bless2011/data_lex_test.tsv', 'r') as f:
    for line in f:
        word1, word2 = line.strip().split('\t')
        test_set.append((word1, word2))

with open('diy.txt', 'w') as f:
    for x in test_set:
        pred = False
        if x in filtered_pairs2paths:
            feature_vector = vectorizer.transform(dict(Counter(filtered_pairs2paths[x]).items()))
            pred = clf.predict(feature_vector)[0]
            if not pred and x in hearst:
                pred = True
            counts[pred] += 1
        else:
            closest = get_closest_pair(filtered_pairs2paths, x)
            if closest:
                feature_vector = vectorizer.transform(dict(Counter(filtered_pairs2paths[closest]).items()))
                pred = clf.predict(feature_vector)[0]
                counts[pred] += 1

        f.write('{}\t{}\t{}\n'.format(x[0], x[1], pred))