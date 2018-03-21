python3 lexicalinference/depPath/extractRelevantDepPaths.py --wikideppaths new_wikipedia_deppaths.txt --trfile bless2011/data_lex_train.tsv --outputfile relevantpaths.txt
python3 lexicalinference/depPath/extractDepPathHyponyms.py --wikideppaths new_wikipedia_deppaths.txt --relevantdeppaths relevantpaths.txt --outputfile extracts.txt

python3 lexicalinference/extractDatasetPredictions.py --extractionsfile extracts.txt --trdata bless2011/data_lex_train.tsv --valdata bless2011/data_lex_val.tsv --testdata bless2011/data_lex_test.tsv --trpredfile deppred/train.txt --valpredfile deppred/val.txt --testpredfile deppred/deppath.txt

python3 lexicalinference/computePRF.py --goldfile bless2011/data_lex_train.tsv --predfile deppred/train.txt
python3 lexicalinference/computePRF.py --goldfile bless2011/data_lex_val.tsv --predfile deppred/val.txt
