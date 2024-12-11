# CSE 143 Final Project

Team Members: Bill Mularski, Navodit Maheshwari, Nikolas Makranyi, Nathan Wu

## Requirements

### pip Packages

The following pip packages are required:
 - requests
 - kagglehub
 - tqdm
 - nltk
 - pandas
 - numpy
 - unidecode
 - gensim

These can be installed by running:
```
pip install requests kagglehub tqdm nltk pandas numpy unidecode gensim
```

### nltk Resources

The following nltk resources are required:
 - punkt_tab
 - averaged_perceptron_tagger_eng
 - wordnet

These can be installed by running:
```
python3
>>> import nltk
>>> nltk.download('punkt_tab')
>>> nltk.download('averaged_perceptron_tagger_eng')
>>> nltk.download('wordnet')
>>> exit()
```

## Running

Before running the main program, please run:
```
python3 download_data.py
```
...and then...
```
python3 extract_data.py
```
...in the main directory if you haven't already.

To run the main program, please run:
```
python3 madlib.py
```
