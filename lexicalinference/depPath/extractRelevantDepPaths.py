import os
import pprint
import argparse
from collections import defaultdict

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--wikideppaths', type=str, required=True)
parser.add_argument('--trfile', type=str, required=True)

parser.add_argument('--outputfile', type=str, required=True)


def extractRelevantPaths(wikideppaths, wordpairs_labels, outputfile):
    '''Each line in wikideppaths contains 3 columns
        col1: word1
        col2: word2
        col3: deppath
    '''

    print(wikideppaths)

    lines_read = 0
    relevantDepPaths2counts = defaultdict(
        lambda: {'forward': 0, 'reverse': 0, 'both': 0, 'negative': 0, 'total': 0})
    with open(wikideppaths, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            lines_read += 1

            word1, word2, deppath = line.split("\t")

            '''
                IMPLEMENT METHOD TO EXTRACT RELEVANT DEPEDENCY PATHS HERE

                Make sure to be clear about X being a hypernym/hyponym.

                Dependency Paths can be extracted in multiple different categories, such as
                1. Forward Paths: X is hyponym, Y is hypernym
                2. Reverse Paths: X is hypernym, Y is hyponym
                3. Negative Paths: If this path exists, definitely not a hyper/hyponym relations
                4. etc......
            '''


            forward_exists = (word1, word2) in wordpairs_labels
            reverse_exists = (word2, word1) in wordpairs_labels

            if forward_exists and reverse_exists \
                and wordpairs_labels[(word1, word2)] \
                and wordpairs_labels[(word2, word1)]:
                relevantDepPaths2counts[deppath]['both'] += 1
                relevantDepPaths2counts[deppath]['total'] += 1
            elif forward_exists and reverse_exists \
                and not wordpairs_labels[(word1, word2)] \
                and not wordpairs_labels[(word2, word1)]:
                relevantDepPaths2counts[deppath]['negative'] += 1
                relevantDepPaths2counts[deppath]['total'] += 1
            elif forward_exists and wordpairs_labels[(word1, word2)]:
                relevantDepPaths2counts[deppath]['forward'] += 1
                relevantDepPaths2counts[deppath]['total'] += 1
            elif reverse_exists and wordpairs_labels[(word2, word1)]:
                relevantDepPaths2counts[deppath]['reverse'] += 1
                relevantDepPaths2counts[deppath]['total'] += 1

    pp.pprint(sorted(relevantDepPaths2counts.items(), key=lambda x: x[1]['total'], reverse=True)[:15])

    with open(outputfile, 'w') as f:
        for dep_path, counts in relevantDepPaths2counts.items():
            if counts['total'] > 5:
                if counts['forward'] / counts['total'] >= .7:
                    f.write(dep_path + '\tForward\n')
                elif counts['reverse'] / counts['total'] >= .8:
                    f.write(dep_path + '\tReverse\n')

def readVocab(vocabfile):
    vocab = set()
    with open(vocabfile, 'r') as f:
        for w in f:
            if w.strip() == '':
                continue
            vocab.add(w.strip())
    return vocab


def readWordPairsLabels(datafile):
    wordpairs = {}
    with open(datafile, 'r') as f:
        inputdata = f.read().strip()

    inputdata = inputdata.split("\n")
    for line in inputdata:
        word1, word2, label = line.strip().split('\t')
        word1 = word1.strip()
        word2 = word2.strip()
        wordpairs[(word1, word2)] = label
    return wordpairs


def main(args):
    print(args.wikideppaths)

    wordpairs_labels = readWordPairsLabels(args.trfile)

    print("Total Number of Word Pairs: {}".format(len(wordpairs_labels)))

    extractRelevantPaths(args.wikideppaths, wordpairs_labels, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
