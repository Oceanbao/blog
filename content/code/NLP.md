---
title: "NLP"
date: 2018-11-25T14:56:55-05:00
showDate: true
draft: false
---

# NLP - Primer

- Text, unstructured particularly, is as aboundant as important to understanding!
- [Introduction to NN Translation with GPUs](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part2/)
- Sources
    - [Open American Natioanl Corpus](http://www.anc.org)
    - [British Natioanl Corpus](http://www.natcorp.ox.ac.uk)
    - [List of Text Corpora](https://en.wikipedia.org/wiki/List_of_text_corpora)
    - [Wikiepedia Dataset](https://en.wikipedia.org/wiki/Wikipedia:Database_download)
    - Twitter (see Text)
- Topic spotting
- Text classification
- Application
    1. chatbot
    2. translation
    3. sentiment analysis



# String Primer

### Unicode
- Remember to include **U** before string to ensure Unicode String

## Regular Expression - strings with special syntax
- allows to match **patterns** in other strings
- Regex, ```import re```
- Matching pattern with string

```python
re.match('abc', 'abcdef')
# word phrase
word_regex = '\w+'
re.match(word_regex, 'hi there!')
```
### Common Regex Patterns
- ```\w+``` [word, 'Magic']
- ```\d``` [digit, 9]
- ```\s``` [space, '']
- ```.*``` [wildcard, 'usename74']
- ```+ or *``` [greedy, 'aaaaa']
- capitalised = Negation, ```\S``` [Not space, 'no_spaces']
- ```[a-z]``` [lowercase group, 'abcedfg']

> return depends, iter, string, or match object

> useful for preprocessing before **TOKENISATION**


```python
my_string = "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"


import re

sentence_endings = r"[.?!]"

print("\n",re.split(sentence_endings, my_string))

capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))


spaces = r"\s+"
print("\n", re.split(spaces, my_string))


digits = r"\d+"
print("\n",re.findall(digits, my_string))
```


     ["Let's write RegEx", "  Won't that be fun", '  I sure think so', '  Can you find 4 sentences', '  Or perhaps, all 19 words', '']
    ['Let', 'RegEx', 'Won', 'Can', 'Or']
    
     ["Let's", 'write', 'RegEx!', "Won't", 'that', 'be', 'fun?', 'I', 'sure', 'think', 'so.', 'Can', 'you', 'find', '4', 'sentences?', 'Or', 'perhaps,', 'all', '19', 'words?']
    
     ['4', '19']


## Tokenisation
- preparing text for NLP
- N-gram, punctuation, hashtages, etc
- NLTK module: ```from nltk.tokenize import word_tokenize```
- **WHY**
    1. easier to map SPEECH PART
    2. matching common words
    3. removing unwanted tokens
    4. e.g. 'I don't like Sam's shoes' -> "I", "do", "n't"...
- Other NLTK class:
    1. ```sent_tokenize`` tokenise document into sentenses
    2. ```regexp_tokenize``` tokenise string or doc on regex pattern
    3. ```TweetTokenizer``` special class for tweet, hashtags, mentions, lots of exclamation points !!!

#### Diff(search, match)
- Search matches all on everywhere, unlike match only onset BUT **useful for entire pattern or beginning**



```python
with open('Data_Folder/TxT/grail_abridged.txt', mode='r') as file:
    scene_one = file.read()

from nltk.tokenize import sent_tokenize, word_tokenize

# Split as sentences
sentences = sent_tokenize(scene_one)

print('\n',sentences[:3])

# tokenise the 4th sentence
tokenize_sent = word_tokenize(sentences[3])

print('\n',tokenize_sent)

unique_tokens = set(word_tokenize(scene_one))

print('\n',unique_tokens)
```


     ['SCENE 1: [wind] [clop clop clop] \nKING ARTHUR: Whoa there!', '[clop clop clop] \nSOLDIER #1: Halt!', 'Who goes there?']
    
     ['ARTHUR', ':', 'It', 'is', 'I', ',', 'Arthur', ',', 'son', 'of', 'Uther', 'Pendragon', ',', 'from', 'the', 'castle', 'of', 'Camelot', '.']
    
     {'they', 'Not', 'husk', 'using', 'fly', 'under', 'tropical', 'That', 'lord', 'point', 'use', 'sovereign', 'winter', 'wants', 'to', 'second', 'Uther', 'ounce', 'together', 'five', 'by', '!', '1', 'coconut', ']', 'and', 'Patsy', 'Saxons', '2', 'join', 'or', 'pound', 'Halt', 'there', 'goes', 'is', 'Yes', 'seek', 'go', 'master', 'do', 'that', 'martin', "'re", 'warmer', 'son', 'ridden', 'SOLDIER', 'with', 'could', ',', "'em", 'Mercea', 'agree', 'mean', 'A', 'KING', 'clop', 'one', 'court', 'You', 'It', 'two', '--', 'get', 'our', 'must', 'They', 'Pendragon', 'forty-three', 'south', 'back', 'bird', 'ratios', '?', 'migrate', 'here', 'breadth', 'will', 'he', 'empty', 'other', 'creeper', "'d", 'the', 'So', 'Will', 'Wait', 'Arthur', 'why', 'got', 'interested', 'have', 'house', 'defeator', 'where', 'Britons', 'dorsal', 'In', 'carried', 'maintain', 'your', 'halves', 'Listen', 'Supposing', 'my', 'length', 'We', 'found', 'temperate', 'Court', 'strangers', 'tell', 'anyway', '[', 'on', 'through', 'Whoa', 'you', 'be', 'England', 'horse', 'yeah', 'am', 'just', "'s", "n't", '#', 'all', 'may', 'African', 'Are', 'land', 'in', 'line', 'Well', 'needs', 'this', 'European', 'Where', 'zone', 'ARTHUR', 'minute', 'castle', 'from', 'Please', 'trusty', 'swallows', "'m", 'then', 'these', 'every', 'simple', 'grip', 'held', 'strand', 'guiding', 'maybe', '...', "'", 'swallow', 'speak', 'weight', 'Found', 'beat', 'me', 'bangin', 'but', 'King', 'What', 'SCENE', 'are', 'kingdom', 'grips', 'velocity', 'suggesting', 'times', 'Pull', 'of', 'Camelot', 'But', 'The', 'question', 'an', 'Ridden', 'snows', '.', "'ve", 'No', 'matter', 'at', 'covered', 'servant', 'who', 'order', 'non-migratory', 'them', 'climes', 'Oh', 'feathers', 'it', 'Am', 'air-speed', 'its', 'plover', 'carrying', 'wings', 'Who', 'wind', 'ask', 'knights', 'I', 'bring', 'if', 'not', 'a', 'does', 'yet', 'course', 'coconuts', 'carry', ':', 'right', 'search', 'sun', 'since'}



```python
# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))
```

    580 588
    <_sre.SRE_Match object; span=(9, 32), match='[wind] [clop clop clop]'>
    <_sre.SRE_Match object; span=(0, 7), match='ARTHUR:'>


## Advanced Tokenisation
- **union** |
- grouping ()
- explicit char range []
- ex:
    1. digit-word = ```('(\d+|\w+)')```
- **range and group**
    - **special character** "\"
    - [A-Za-z]+ "upper/lower alphabet"
    - [0-9] "numeric 0-9"
    - [A-Za-z\-\.]+ "upper/lower, - and ."
    - (a-z) "a - and z" **GROUP**
    - (\s+|,) "spaces or comma"
- e.g. '[a-z0-9 ]+' meaning lower, digits, SPACE, greedily -> using .match() will **stop at any punctuation, as not specified**



```python
my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"

# to tokenise by words & punctuation & retain #1

from nltk.tokenize import regexp_tokenize

regexp_tokenize(my_string, '(\w+|#\d|\?|!)')
```




    ['SOLDIER',
     '#1',
     'Found',
     'them',
     '?',
     'In',
     'Mercea',
     '?',
     'The',
     'coconut',
     's',
     'tropical',
     '!']




```python
# Tweet tokenisation
tweets = ['This is the best #nlp exercise ive found online! #python',
 '#NLP is super fun! <3 #learning',
 'Thanks @datacamp :) #nlp #python']

from nltk.tokenize import TweetTokenizer

# first use regexp via key symbols

print(regexp_tokenize(str(tweets), '[@#]\w+'))

# then specialised class

tweetTK = TweetTokenizer()

all_tokens = [tweetTK.tokenize(t) for t in tweets]

print('\n', all_tokens)
```

    ['#nlp', '#python', '#NLP', '#learning', '@datacamp', '#nlp', '#python']
    
     [['This', 'is', 'the', 'best', '#nlp', 'exercise', 'ive', 'found', 'online', '!', '#python'], ['#NLP', 'is', 'super', 'fun', '!', '<3', '#learning'], ['Thanks', '@datacamp', ':)', '#nlp', '#python']]



```python
# Non-ASCII tokenisation

german_text = 'Wann gehen wir Pizza essen? 🍕 Und fährst du mit Über? 🚕'

from nltk.tokenize import word_tokenize

word_tokenize
```




    <function nltk.tokenize.word_tokenize(text, language='english', preserve_line=False)>



#### Unicode ranges for emoji are:
- ('\U0001F300'-'\U0001F5FF')
- ('\U0001F600-\U0001F64F')
- ('\U0001F680-\U0001F6FF')
- ('\u2600'-\u26FF-\u2700-\u27BF')


```python
all_words = word_tokenize(german_text)
print(all_words)

capital_german = r"[A-ZÜ]\w+"
print('\n', regexp_tokenize(german_text, capital_german))

emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"

print('\n', regexp_tokenize(german_text, emoji))
```

    ['Wann', 'gehen', 'wir', 'Pizza', 'essen', '?', '🍕', 'Und', 'fährst', 'du', 'mit', 'Über', '?', '🚕']
    
     ['Wann', 'Pizza', 'Und', 'Über']
    
     ['🍕', '🚕']


## Charting Word Length
- Charting and graphs and animations
- Visualise


```python
with open('Data_Folder/TxT/grail.txt', mode='r') as file:
    grail_text = file.read()

grail_lines = grail_text.split(sep='\n')

# replacing all script lines for speaker or NAME: instances
pattern_speaker = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"

grail_lines = [re.sub(pattern_speaker, '', line) for line in grail_lines]

grail_tokenised = [regexp_tokenize(s, "\w+") for s in grail_lines]

line_num_words = [len(t_line) for t_line in grail_tokenised]

plt.hist(line_num_words)
plt.show()
```




    (array([916., 177.,  52.,  22.,   9.,   4.,   4.,   5.,   1.,   2.]),
     array([  0. ,  10.3,  20.6,  30.9,  41.2,  51.5,  61.8,  72.1,  82.4,
             92.7, 103. ]),
     <a list of 10 Patch objects>)




![png](output_14_1.png)


# Bag-of-word Counting
- Tokenise-Count flow
- frequency is a common statistics (max or min)
- e.g. lower(all text) to evade duplication
- e.g. preprocess by expurgating articles and frivolous words


```python
with open('Data_Folder/TxT/wiki_txt2/wiki_text_debugging.txt', 'r') as file:
    wiki_debugging = file.read()

from collections import Counter

wiki_debugging_TK = word_tokenize(wiki_debugging)

wiki_debugging_lower = [t.lower() for t in wiki_debugging_TK]

bow_wiki_debugging = Counter(wiki_debugging_lower)

print(bow_wiki_debugging.most_common(10))

type(bow_wiki_debugging)
```

    [(',', 151), ('the', 150), ('.', 89), ('of', 81), ("''", 66), ('to', 63), ('a', 60), ('``', 47), ('in', 44), ('and', 41)]





    collections.Counter



## Text Preprocessing
- preparing for ML or analysis
- e.g. token, bow, lowercasing, etc.
- **Lemmatisation/Stemming** shortening to root stems
    - `.isalpha()`

    - `from ntlk.corpus import stopwords`
      `[... if t not in stopwords.words('english')]`
- try and error method to context




```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

alpha_only = [t for t in wiki_debugging_lower if t.isalpha()]

english_stops = stopwords.words('english')

no_stops = [t for t in alpha_only if t not in english_stops]

wordnet_lemmatizer = WordNetLemmatizer()

lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

bow = Counter(lemmatized)

print(bow.most_common(10))
```

    [('debugging', 40), ('system', 25), ('bug', 17), ('software', 16), ('problem', 15), ('tool', 15), ('computer', 14), ('process', 13), ('term', 13), ('debugger', 13)]


## GENSIM
- open NLP lib using top academic models to action complex tasks
    - Vectorising doc or word
    - Topic spotting and comparison
- **vector space**, **distance**, **similarity** analysis for semantics and relation
- Typically SparseMatrix format, [muse]
- Example
    - length(Male-Female) ~= length(King-Queen)
    - length-diff(verb tenses)
    - Node and Edge of capital-city pair
- In a nutshell, gensim level up token-id-count process of doc or word
    - `from gensim.corpora.dictionary import Dictionary`

    - doc = ['list of strings']
    - tokenise doc.lower
    - `dict = Dictionary(doc_tk)` 
      `dict.token2id` 
      a {} of all token-ID allocation

    - build own corpus `[dict.doc2bow(doc) for doc in doc_tk]`
    - corpus is now [[(id, count)],[]..] 
- **Key**
    1. ease of storing, updating and reuse
    2. dict is mutable
    3. more advanced and feature rich BOW to used 


```python
# loop open files in filepath + pattern

import os
import glob

article_sum = []

for filepath in glob.glob(os.path.join('Data_Folder/TxT/wiki_txt2/', '*.txt')):
    with open(filepath,'r') as file:
        article_raw = file.read()
        article_lower = [t.lower() for t in word_tokenize(article_raw)]
        article_alpha = [t for t in article_lower if t.isalpha()]
        article_tk = [t for t in article_alpha if t not in english_stops]
        article_sum.append(article_tk)
```


```python
len(article_sum)
for i in range(len(article_sum)):
    print(len(article_sum[i]))
```




    12



    4095
    2894
    1051
    2963
    6947
    1984
    463
    3235
    2109
    1271
    722
    3045



```python
from gensim.corpora.dictionary import Dictionary

dictionary = Dictionary(article_sum)

computer_id = dictionary.token2id.get("computer")

print(dictionary.get(computer_id))

corpus = [dictionary.doc2bow(article) for article in article_sum]

print(corpus[4][:10]) # 5th doc first 10 mots
```

    computer
    [(4, 1), (6, 6), (7, 2), (9, 5), (18, 1), (19, 1), (20, 1), (22, 1), (24, 2), (28, 3)]



```python
doc_fifth = corpus[4]

doc_fifth[:4] # List of tuples (id, count)

# sort doc by frequency, or count
bow_doc_fifth = sorted(doc_fifth,
                      key = lambda w: w[1],
                      reverse=True)

# print top-5 (id + count)
for id, count in bow_doc_fifth[:5]:
    print(dictionary.get(id), count)

from collections import defaultdict

total_word_count = defaultdict(int)

import itertools

for id, count in itertools.chain.from_iterable(corpus):
    total_word_count[id] += count

sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)

for id, count in sorted_word_count[:5]:
    print(dictionary.get(id), count)

```




    [(4, 1), (6, 6), (7, 2), (9, 5)]



    computer 251
    computers 100
    first 61
    cite 59
    computing 59
    computer 597
    software 450
    cite 322
    ref 259
    code 235


# Tf-idf + Gensim
- **Term Frequency - Inverse Document Frequency**
- Common model used to id importance in each document **from the corpus**
- Logic: each corpus may have shared words beyond just stopwords
    - down-weighted importance of context-word
    - e.g. astronomy: 'Sky'
    - dismissing context-adjusted words
    - up-weighted specific frequency
- Function
### $w_{i,j} = tf_{i,j} * \log(\frac{N}{df_i})$
    - $w_{i,j}$ = tf-idf weight for token i in doc j
    - $tf_{i,j}$ = # occurences of token i in doc j
    - $df_i$ = # doc containing token i
    - N = total # of doc
- E.g. 'computer' appears 5 times in a doc of 100 words; what's weight given a corpus of 200 doc of which 20 doc mentioned the word
    - (5/100) * log(200/20) 
    - tf = percentage share of word compared to all tokens in the doc idf = log(total doc / contained doc)


```python
from gensim.models.tfidfmodel import TfidfModel

tfidf = TfidfModel(corpus)

# strange apply of TfidfModel() instance
tfidf_weights = tfidf[doc_fifth]

print(tfidf_weights[:5],'\n')

# sort weights highest to lowest
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

for id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(id), weight)
```

    [(4, 0.005117037137639146), (6, 0.005095225240405224), (7, 0.00815539037400173), (9, 0.02558518568819573), (18, 0.003228490980266662)] 
    
    mechanical 0.1836016185939278
    circuit 0.15046224827624213
    manchester 0.14187397800439872
    alu 0.13888822917806967
    thomson 0.12731421007989718


# Named Entity Recognition
- Motif: NLP task to identify important NE in the text
    1. people, places, organisations
    2. dates, states, nouns
- Used alongside topic identification
## Stanford CoreNLP Library
- Integrated into Python via nltk
    - Java based
    - API able
    - Support for NER + coReference and dependency trees
- Built-in pos_tag, etc in nltk


```python
with open('Data_Folder/TxT/News articles/uber_apple.txt','r') as file:
    article_uber = file.read()

import nltk

# first tk article into sentences
sentence_uber = sent_tokenize(article_uber)

# second tk sentences into words
sentence_uber_tk = [word_tokenize(sent) for sent in sentence_uber]

# tag words into speech-part 
pos_sentences = [nltk.pos_tag(sent) for sent in sentence_uber_tk]

# Create Named Entity chunks
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of Tree with 'NE' tags

for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)


```

    (NE Uber/NNP)
    (NE Beyond/NN)
    (NE Apple/NNP)
    (NE Uber/NNP)
    (NE Uber/NNP)
    (NE Travis/NNP Kalanick/NNP)
    (NE Tim/NNP Cook/NNP)
    (NE Apple/NNP)
    (NE Silicon/NNP Valley/NNP)
    (NE CEO/NNP)
    (NE Yahoo/NNP)
    (NE Marissa/NNP Mayer/NNP)



```python
# Charting NE

ner_categories = defaultdict(int)

# loading a new article in non-binary form
with open('Data_Folder/TxT/News articles/articles.txt', 'r') as file:
    article_nonBinary = file.read()

chunked_nonBinary = nltk.ne_chunk_sents(
    [nltk.pos_tag(sent) for sent in 
     [word_tokenize(sent) for sent in 
      sent_tokenize(article_nonBinary)]], binary=False)

for sent in chunked_nonBinary:
    for chunk in sent:
        if hasattr(chunk, "label"):
            ner_categories[chunk.label()] += 1

labels = list(ner_categories.keys())

values = [ner_categories.get(l) for l in labels]

plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

plt.show()
```




    ([<matplotlib.patches.Wedge at 0x1a2bd896a0>,
      <matplotlib.patches.Wedge at 0x1a2bd89da0>,
      <matplotlib.patches.Wedge at 0x1a2bd8e4e0>,
      <matplotlib.patches.Wedge at 0x1a2bd8ebe0>,
      <matplotlib.patches.Wedge at 0x1a2bd58320>],
     [Text(-1.07181,0.247446,'ORGANIZATION'),
      Text(0.341578,-1.04562,'GPE'),
      Text(0.201317,1.08142,'PERSON'),
      Text(-0.811594,0.742506,'LOCATION'),
      Text(-0.832466,0.719027,'FACILITY')],
     [Text(-0.584622,0.134971,'15.0%'),
      Text(0.186315,-0.570339,'52.3%'),
      Text(0.109809,0.589866,'31.8%'),
      Text(-0.442688,0.405003,'0.5%'),
      Text(-0.454073,0.392197,'0.5%')])




![png](output_28_1.png)


# SpaCy - NLP library similar to Gensim 
- Focus on creating pipeline to generate models and corpora
- Focus on GTD not academic - ONLY ONE NER per langue, no-frills!
- So what's **academic** stuff? 
    - Focus not Edge-Cutting algo, **NLTK** focus on giving scholars toolkit to play around
- FEATURES OF SPACY
    1. Non-destructive tokenisation
    2. 21+ langues
    3. 6 statsmodels for 5 langues
    4. pre-trained WordVEC
    5. esy DeepLearning integration
    6. POS tagging
    7. NER
    8. Labeled DEPENDENCY parsing
    9. Syntax-drven sentence segmentaion
    10. Built-in visual for syntax and NER
    11. easy string-to-hash mapping
    12. export to NPArray
    13. Efficient binary SERIALISATION
    14. easy model pkg and deployment
    15. Speed
    16. Robust, rigorously evaluated accuracy
- e.g. Visualiser online by `Displacy`
- NER to load
- Including advanced German and Chinese
## Why SpaCy for NER
- easy pipelineing 
- different entity types 
- informal corpora - tweets chat
- quicly growing


## Language Model - stats model for NLP tasks
- need download separately
- language specific
- installation

```bash
spacy download en / de / es / fr / xx # multi-langue
spacy download en_core_web_sm # best mactching version of specific model for spacy
```

- Loading language model

```python
import spacy
nlp = spacy.load('en')
# process text via pipeline
doc = nlp(u'This is a sentence.') # u can be ignore in Python3
```

### Basic Cleaning by SpaCy
> calling above on U-text spaCy first TOKENISES text to make a Doc Project, then processed in 7 steps 
1. tokenizer
2. tensoriser
3. tagger
4. parser
5. NER
6. output Doc

### Tokenizing Text
- splitting text into meaningful tokens/parts - words, puncs, numbers, special char, building blocks
- More than .split() - syntax-aware split, don't or U.K. 
    - "don't" -> 2 token {ORTH: "do"} and {"ORTH": "n't", LEMMA: "not"}
    - ORTH refers to textual content, LEMMA the word with no information suffix
- Creating own Tokenisers in [Linguisitic Features](https://spacy.io/usage/linguistic-features#section-tokenization)
- Once done, Doc obj comprising tokens each is then worked on by other components of the PIPELINE

### POS tagging - **tensorizer**
- encode internal repr of doc as ARRAY of floats, necessary for NN need tensors
- Mark token of sentence with proper part of speech
- use Stats Models to perform POS tagging

```python
for token in doc:
    token.text, token.pos_
```

### Dependency Parsing
- while parsing refers to any analysis of string of symbols to understand relationship, dependency parsing emphasises on DEPENDENCY
- Subject or Object Noun

### NER
- real-world object with name
- spacy built-in training but may need tuning/training

```python
for ent in doc.ents:
    ent.text, ent.start_char, ent.end_char, ent.label_
```

- Built-int NE types
  **PERSON, NORP, FACILITY, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE**

### Rule-based matching
**ORTH, LOWER,UPPER, IS_ALPHA, IS_ASCII, IS_DIGIT, IS_PUNCT, IS_SPACE, IS_STOP, LIKE_NUM, LIKE_URL, LIKE_EMAIL, POS, TG, DEP, LEMMA, SHAPE**
- default pipeline perform further annotating tokens with more info 
- self-defined rule is possible

### Preprocessing
- stop-word by `token.IS_STOP` attribute boolean 
- self-defined stoppers

```python
my_stops = ['say', 'be', 'said', 'says', 'saying', 'field']
for stopword in my_stops:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True

# alternatively
from spacy.lang.en.stop_words import STOP_WORDS

print(STOP_WORDS)
STOP_WORDS.add("additioanl here")
```

- STEMMING & LEMMATISATION
    - Stemming often chops off end of word following basic rules / CONTEXTLESS not POS-based
    - Lemmatisation however conducts MORPHOLOGICAL analysis to find root word
    - Stanford NLP book explains
    - lemma accessed `.lemma_` attribute

- Basic cleaning

```python
doc = nlp('some text')
sentence = []
for w in doc:
    if w.text != 'n' and not w.is_stop and not w.is_punct and not w.like_num:
        sentence.append(w.lemma_)
print(sentence)
```

- removing trash NOTR appending lemmatised form of word !!
- further remove based on need, e.g. **removing all VERB via checking POS tag of token !**

> RECAP: spaCy pipeline annotates text easily to retain info to process analysis, always first starting task in NLP
1. possible annotating text with LOTs of info (tokenisation, stoppers, POS, NER, etc)
2. possible TRAINING annotationg models on own, power to language models and processing pipeline!


## GENSIM - Vectorising Text and N-grams
- VECTORISING TEXT - BOW, TF-IDF, LSI (latent semantic indexing), WORD2VEC
> GENSIM included novel kits like LDA (Latent Dirichlet allocation), Latent Semantic Analysis, Random projection, Hierarchical Dirichlet process, word2vec deep learning, cluster computing
- Memory-efficient, scalable (generators/iterators, most IR algo ~ Matrix Decompo)
- [Documentation](https://github.com/RaRe-Technologies/gensim/tre/develop/docs/notebooks)

## Vectorised Word
### BOW - most straightforward
- Word-Freq mapped against VOCAB
- **OrderLESS** NO **Spatial INFO** - or semantics
- BUT many cases no need for Spatial INFO in Information Retrieval Algo
- EX: spam filtering via **Naive Bayes Classifier**

### TF-IDF
- Largely used in search engines to find relevant docs based on query
- Weigthed Frequency against Occurences

### Other Forms
- Topic Models
- [The Amazing Power of Word Vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)

### Vectorisation in GENSIM

```python
from gensim import corpora
documents = [u'list of text']
import spacy
nlp = spacy.load('en')
texts = []
for document in documents:
    text = []
    doc = nlp(document)
    for w in doc:
        if not w.is_stop and not w.is_punct and not w.like_num:
            text.append(w.lemma_)
    texts.append(text)
print(texts)

# whipping up BOW for mini-corpus
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]
```

- A List of List each repr BOW `(word_id, word_count)` a tuple
- NOTE
    1. Above case is fully RAM-loaded, in production need to store corpus
    ```python
        corpora.MmCorpus.serialize('/tmp/example.mm', corpus)
    ```
    2. Thus stored on disk away from RAM, at most one vector resides in RAM ([tutorial](https://radimrehurek.com/gensim/tut1.html))
    3. From BOW into TF-IDF: tfidf is a table TRAINED on own corpus, involving simply going through supplied corpus \n once and computing df on all features (other models, LSA or LDA, much more invovled)
    ```python
        from gensim import models
        tfidf = models.TfidfModel(corpus)
        # Check the result
        for document in tfidf[corpus]:
            print(document)
    ```
    4. Score (0,1) measuring importance of word in corpus - used in ML models or further chain/link vectors by performing other transformation on them

#### N-gram plus more preprocessing
- Calculated by conditional proba around words called **collocation** based on corpus provided
- This extra preprocess precedes **DOC2BOW** dictionary!

```python
bigram = gensim.models.Phrases(texts) # resulting a trained bi-gram model for corpus, then transformation on new text
texts = [bigram[line] for line in texts] # each line having all possible bi-grams created
```

### RECAP
- Preprocess can be as complex as need dictates
- EX Removing both high-freq and low-freq words (GENSIM `dictionary` module) 
> Rid of occurence < 20 documents, or in > 50% of documents
```python
dictionary.filter_extremes(no_below=20, no_above=0.5)
```
- EX remove most-freq tokens or prune out certain token IDs [example](https://radimrehurek.com/gensim/corpora/dictionary.html)



## POS-Tagging and Applications
- What
    1. Not possible to tag POS unless in sentence or phrase
    2. SpaCy has 19 POS tags `.tag_` attr and `.pos_`
    3. Brown Corpus used HMM to predict tags (HMM - sequentail model)
- Current
    1. Stats model and Deep Learning [ACL list of results](https://aclweb.org/aclwiki/POS_Tagging_(State_of_the_art))
    2. spaCy early tagger is **averaged perceptron**
- Why
    1. Historically speech-to-text conversion / translation disambiguate homonyms
    2. Dependency parsing
    3. Demo by [SpaCy Display](https://explosion.ai/demos/display)
- PYTHONIC
    1. NLTK the main rival POS-tagger
    ```python
        import nltk
        text = nltk.word_tokenize("And now for something completely different")
        nltk.pos_tag(text)
        bigram_tagger = nltk.BigramTagger(train_sents) # one of many tagger options in NLTK
        bigram_tagger.tag(text)
    ```
    2. Resources [Official Doc of tag module](https://www.nltk.org/api/nltk.tag.html), [NLTK book](https://www.nltk.org/book/ch05.html), [Training POS](https://textminingonline.com/dive-into-nltk-part-iii-part-of-speech-tagging-and-pos-tagger)
    3. Other Modules [AI in Practice: Identifying Parts of Speech in Python](https://medium.com/@brianray_7981/ai-in-practice-identifying-parts-of-speech-in-python-8a690c7a1a08)
        - TEXTBLOB is likely the ONLY other POS tagger worth a look, performing similarly to one in SpaCy - algo written by spaCy maintainer [Detail](https://stevenloria.com/pos-tagging/)

#### POS-Tagging in SpaCy (97% accuracy battery-packed)
- Built-in tagging in PIPELINE (from spacy.load('en') and loading text, `for token in sent: token.text, token.pos_, token.tag_`
- EX **fishy** a tricky word possible multi-POS but correctly machine-learned by multiple FEATURES (e.g. surrending POS, suffix-prefix, etc.)
- SpaCy returns **kills many matephorical birds** with the same stone! 

#### Adding Defined Training Models
- Probabilistically improvable to relevant context and data - [Official SpaCy Traininng process](https://spacy.io/usage/training)
- Training Loop:
    1. Provide text + part to train (entities, heads, deps, tags, cats)

```python
TRAIN_DATA = [
    ("Facebook has been accused for leaking personal data of users.", {"entities": [(0, 8, 'ORG')]}),
    ...] # Facebook is the entity marked as ORG (start_index, end_index)
nlp = spacy.blank('en')
optimizer = nlp.begin_training()
for i in range(20):
    random.shuffle(TRAIN_DATA)
    for text, annotations in TRAIN_DATA:
        nlp.update([text], [annotateions], sgd = optimizer)
nlp.to_disk('/model')
```

- Trainng POS-tagger [example code](https://github.com/explosion/spacy/blob/master/examples/training/train_tagger.py)
    1. init dictionary, define mapping from data's POS to [Universsal POS tag set](http://universaldependencies.org/docs/u/pos/index.html)
    2. More data better accuracy...

```python
# nltk for POS-tagging

import nltk
text = word_tokenize("And now for something completely different")
nltk.pos_tag(text)

bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(text)

import spacy
nlp = spacy.load('en')
sent_0 = nlp(u'Mathieu and I went to the park.')
sent_1 = nlp(u'If Clement was asked to take out the garbage, he would refuse.')
sent_2 = nlp(u'Baptiste was in charge of the refuse treatment center.')
sent_3 = nlp(u'Marie took out her rather suspicious and fishy cat to go fish for fish.')

for token in sent_0:
    print(token.text, token.pos_, token.tag_)

for token in sent_1:
    print(token.text, token.pos_, token.tag_)

for token in sent_2:
    print(token.text, token.pos_, token.tag_)

for token in sent_3:
    print(token.text, token.pos_, token.tag_)

# training NER

TRAIN_DATA = [
     ("Facebook has been accused for leaking personal data of users.", {'entities': [(0, 8, 'ORG')]}),
     ("Tinder uses sophisticated algorithms to find the perfect match.", {'entities': [(0, 6, "ORG")]})]

nlp = spacy.blank('en')
optimizer = nlp.begin_training()
for i in range(20):
    random.shuffle(TRAIN_DATA)
    for text, annotations in TRAIN_DATA:
        nlp.update([text], [annotations], sgd=optimizer)
nlp.to_disk('/model')



## run this code as a seperate file

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# You need to define a mapping from your data's part-of-speech tag names to the
# Universal Part-of-Speech tag set, as spaCy includes an enum of these tags.
# See here for the Universal Tag Set:
# http://universaldependencies.github.io/docs/u/pos/index.html
# You may also specify morphological features for your tags, from the universal
# scheme.
TAG_MAP = {
    'N': {'pos': 'NOUN'},
    'V': {'pos': 'VERB'},
    'J': {'pos': 'ADJ'}
}

# Usually you'll read this in, of course. Data formats vary. Ensure your
# strings are unicode and that the number of tags assigned matches spaCy's
# tokenization. If not, you can always add a 'words' key to the annotations
# that specifies the gold-standard tokenization, e.g.:
# ("Eatblueham", {'words': ['Eat', 'blue', 'ham'] 'tags': ['V', 'J', 'N']})
TRAIN_DATA = [
    ("I like green eggs", {'tags': ['N', 'V', 'J', 'N']}),
    ("Eat blue ham", {'tags': ['V', 'J', 'N']})
]


@plac.annotations(
    lang=("ISO Code of language to use", "option", "l", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(lang='en', output_dir=None, n_iter=25):
    """Create a new model, set up the pipeline and train the tagger. In order to
    train the tagger with a custom tag map, we're creating a new Language
    instance with a custom vocab.
    """
    nlp = spacy.blank(lang)
    # add the tagger to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    tagger = nlp.create_pipe('tagger')
    # Add the tags. This needs to be done before you start training.
    for tag, values in TAG_MAP.items():
        tagger.add_label(tag, values)
    nlp.add_pipe(tagger)

    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            nlp.update([text], [annotations], sgd=optimizer, losses=losses)
        print(losses)

    # test the trained model
    test_text = "I like blue eggs"
    doc = nlp(test_text)
    print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the save model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])


if __name__ == '__main__':
    plac.call(main)

    # Expected output:
    # [
    #   ('I', 'N', 'NOUN'),
    #   ('like', 'V', 'VERB'),
    #   ('blue', 'J', 'ADJ'),
    #   ('eggs', 'N', 'NOUN')
    # ]
```

- For real-case, training data massisve and assembling it a huge part of training work
- This case the ML model abstracted as `update()` only that it works well and a NN
- For advanced this [blog](https://nlpforhackers.io/training-pos-tagger/) offers NLTK process using self-defined classifiers from SKL
- The SpaCy POS training model [A Good Part-Of-Speech Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python) same as Textblob

#### Small Examples of POS usage
- Change all Verbs to uppercase
    ```python
    def make_verb_upper(text, pos):
        return text.upper() if pos == "VERB" else text
    doc = nlp(u'Tom ran swiftly and walked slowly')
    text = ''.join(make_verb_upper(w.text_with_ws, w.pos_) for w in doc)
    print(text)
    ```
- Count of POS
    ```python
    harry_potter = open("HP1.txt").read()
    hp = nlp(harry_potter)
    hpSents = list(hp.sents)
    hpSentenceLenghts = [len(sent) for sent in hpSents]
    [sent for sent in hSents if len(sent) == max(hpSentencLenghts)]
    hpPOS = pd.Series(hp.count_by(spacy.attrs.POS)) / len(hp)
    
    tagDict = {w.pos: w.pos_ for w in hp}
    hpPOS = pd.Series(hp.count)By(spacy.attrs.POS)) / len(hp)
    df = pd.DataFrame([hpPOS]], index = ['Harry Potter'])
    df.columns =  [tagDict[column] for column in df.columns]
    df.T.plot(kind='bar')
    
    # most common pronoun
    hpAdjs = [w for w in hp if w.pos_ == 'PRON']
    Counter([w.string.strip() for w in hpAdjs]).most_common(10)
    ```
> **Knowldge of POS-tags gives more in-depth text analysis, a pillar of NLP, and after tokenzisng text often the first piece of analysis**


## NER-Tagging and Applications
- Real-world object **GPE, PER, ORG** etc
- **Named Entity Disambiguation NED**
- Unlike POS-tagging, NER-tagging is **CONTEXT AND DOMAIN BASED**
- **CONDITIONAL RANDOM FIELDS** OFTEN USED TO TRAIN NER-TAGGER [CRF: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cgi/viewcontent.cig?referer=&httpsredir=1&article=1162&context=cis_papers)

### NER-Tagging in SpaCy (skipped NLTK case)
> **the power of SpaCy battery-packed pipeline when loading pre-trained model, all of the above mentioned + dependency parsing are produced from that single method spacy.load() - POS, NER**

```python
for token in sent_0: # after nlp()
    token.text, token.ent_type_
# recall SpaCy intends user to access entities in doc.ents streamable object - slice of Doc class is called Span class
for ent in sent_0.ents:
    ent.text, ent.label_
```

#### NER-Tagging Training in SpaCy
```python
# nltk for NER-tagging

from nltk.chunk import conlltags2tree, tree2conlltags

sentence = "Clement and Mathieu are working at Apple."
ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
 
iob_tagged = tree2conlltags(ne_tree)
print iob_tagged

ne_tree = conlltags2tree(iob_tagged)
print ne_tree

from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',  '/usr/share/stanford-ner/stanford-ner.jar', encoding='utf-8')

st.tag(‘Baptiste Capdeville is studying at Columbia University in NY’.split())

import spacy
nlp = spacy.load('en')

sent_0 = nlp(u'Donald Trump visited at the government headquarters in France today.')

sent_1 = nlp(u'Emmanuel Jean-Michel Frédéric Macron is a French politician serving as President of France and ex officio Co-Prince of Andorra since 14 May 2017.')

sent_2 = nlp(u'He studied philosophy at Paris Nanterre University, completed a Master’s of Public Affairs at Sciences Po, and graduated from the École nationale d\'administration (ÉNA) in 2004. ')

sent_3 = nlp(u'He worked at the Inspectorate General of Finances, and later became an investment banker at Rothschild & Cie Banque.')

for token in sent_0:
    print(token.text, token.ent_type_)

for ent in sent_0.ents:
    print(ent.text, ent.label_)

for token in sent_1:
    print(token.text, token.ent_type_)

for ent in sent_1.ents:
    print(ent.text, ent.label_)

for token in sent_2:
    print(token.text, token.ent_type_)

for ent in sent_2.ents:
    print(ent.text, ent.label_)

for token in sent_3:
    print(token.text, token.ent_type_)

for ent in sent_3.ents:
    print(ent.text, ent.label_)
```

#### 2 Training Example: Blank model and Adding New Entity
> The same princiles as POS-tagger
1. Start by adding NER label to pipeline
2. Disabling all other components of PIPI so that only train/update NER-Tagger
3. Training is backend, API by nlp.update()

```python

# run this code seperately

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# training data
TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    plac.call(main)

    # Expected output:
    # Entities [('Shaka Khan', 'PERSON')]
    # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
    # ('Khan', 'PERSON', 1), ('?', '', 2)]
    # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
    # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3),
    # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]
```

#### Adding New Class to Model
> Same principle
1. Load model, disable PIPE not updating
2. Add new label, then loop over the examples and update them
3. Actual training is performed by looping over the examples and calling `nlp.entity.update()`
4. `update()` predict each word then consults annotations provided on `GoldParse` instance to check
5. If wrong, adjusting weight to correct 

```python
# run this code seperately:

"""Example of training an additional entity type
This script shows how to add a new entity type to an existing pre-trained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.
The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.
After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# new entity label
LABEL = 'ANIMAL'

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    ("Horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("Do they bite?", {
        'entities': []
    }),

    ("horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("horses pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("they pretend to care about your feelings, those horses", {
        'entities': [(48, 54, 'ANIMAL')]
    }),

    ("horses?", {
        'entities': [(0, 6, 'ANIMAL')]
    })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, new_model_name='animal', output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()



    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = 'Do you like horses?'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)
```

> The rest code remains the same logic, CRUCIAL DIFFERENCE is training data, ADDING NEW CLASS, and considering need to ADD OLDER EXAMPLES TOO
1. [spaCy's NER linguistic features](https://spacy.io/usage/training#section-ner) with useful advice on HOW TO SET ENTITY ANNOTATIONS
2. spaCy main PIPELINE, offers customisation at each STEP
3. Backend is statistical model accepting FEATURES making PRED
4. There're tuto on how to build CLASSIFIER or update NLTK clf
5. [complete guide to building own NER](https://nlpforhackers.io/named-entity-extraction/), [Intro to NER](https://depends-on-the-definition.com/introduction-named-entity-recognition-python/), [Performing Sequence Labelling using CRF](www.albertauyeung.com/post/python-sequence-labelling-with-crf/)

### VISUALISATION


### DEPENDENCY PARSING
- Parsing is possible on any form/kind with formal GRAMMAR
- 2 Kinds TRADITIONAL UNDERSTANDING vs COMPUTATIONAL LINGUISTICS formal analysis by algo in PARSE TREE
- 2 SCHOOLS IN TRADITIONAL
    1. Dependency Parsing VS Phrase Structure Parsing
> DP is new approach credited to French linguist Lucien Tesnière
1. Constituency Parsing on the other hand is older to Aristole's ideas on term logic.
2. Formally credited to Noam Chomsky, father of linguistics
3. IDEA: words in sentences are connected to each other with directed links - info about relationship between words
4. IDEA: phrase structure parsing, break up into prahses or constituents - grouping sentences
5. Spacy uses SYNTACTIC PARSING

##### Dependency Parsing in Python (NLTK)
- NLTK provides most options in parsing methods, BUT forced to pass own GRAMMAR for effective results
- Not purpose to learn grammars before run compLing algo
- Demo below is how **Stanford Dependency Parser** wrapped NLTK
- First step to download necessary JAR files [Historical Stanford Statistical Parser](https://nlp.stanford.edu/software/lex-parser.shtml)

```python
# NLTK example, be sure to download JAR

from nltk.parse.stanford import StanfordDependencyParser
path_to_jar = 'path_to/stanford-parser-full-2014-08-27/stanford-parser.jar'
path_to_models_jar = 'path_to/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

result = dependency_parser.raw_parse('I shot an elephant in my sleep')
dep = result._next_()
list(dep.triples())
```

#### Dependency Parsing in SpaCy
- Parsing part of PIPELINE does both PHRASAL and DEPENDENCY parsing - able to get info about what NOUN and VERB chunks in sentence are as well as info on dependencies between words
- **Phrasal parsing can also be referred to as chunking, part of sentences or phrases `noun_chunks` attribute**

```python

# spaCy

import spacy
nlp = spacy.load('en')

sent_0 = nlp(u'Myriam saw Clement with a telescope.')
sent_1 = nlp(u'Self-driving cars shift insurance liability toward manufacturers.')
sent_2 = nlp(u'I shot the elephant in my pyjamas.')

for chunk in sent_0.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)

for chunk in sent_1.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)

for chunk in sent_2.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)

for token in sent_0:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])

for token in sent_1:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])

for token in sent_2:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])

from spacy.symbols import nsubj, VERB
# Other ways navigating tree - identify one head per sentence via iterating possible subjects instead of verbs

verbs = set()
for possible_subject in sent_1:
    if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
        verbs.add(possible_subject.head)

# Iterated through all words and checked cases where a nomial subject and head is verb

# possibel to search verbs directly but by double-iteration
verbs = []
for possible_verb in doc:
    if possible_verb.pos == VERB:
        for possible_subject in possible_verb.children:
            if possible_subject.dep == nsubj:
                verbs.append(possible_verb)
                break

# Also useful ATTR lefts, rights, n_rigths ,n_lefts giving info about particular token in tree
# example to finding phrases using syntactic head

root = [token for token in sent_1 if token.head == token][0]
subject = list(root.lefts)[0]
for descendant in subject.subtree:
    assert subject is descendant or subject.is_ancestor(descendant)
    print(descendant.text, descendant.dep_, descendant.n_lefts, descendant.n_rights,
          [ancestor.text for ancestor in descendant.ancestors])

# Find root by seeing where head is token-itself. Subject to the left of tree, iterate subject priting descenddants and number of leaves

# above more realistic finding commonly used ADJ to describe a character in a book
adjectives = []
for sent in book.sents: 
    for word in sent: 
        if 'Character' in word.string: 
            for child in word.children: 
                if child.pos_ == 'ADJ': adjectives.append(child.string.strip())
Counter(adjectives).most_common(10)
```

- Code remains simple but effective - iterating over books sentences, looking for character of sentence, children of character
- Then check if child is an ADJ. Being child means likely marked as DEP, with root word (i.e. Character) described by child
- By checking most common ADJ to mini-analyse characters of books

#### Training DEP Parsers
- Again, SpaCy abstruct the hardest part of ML - **selecting features**
- All left to do is inputting proper training data and set up API to update models

```python

# run the next code as a seperate file

"""Example of training spaCy dependency parser, starting off with an existing
model or a blank model. For more details, see the documentation:
* Training: https://spacy.io/usage/training
* Dependency Parse: https://spacy.io/usage/linguistic-features#dependency-parse
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# training data
TRAIN_DATA = [
    ("They trade mortgage-backed securities.", {
        'heads': [1, 1, 4, 4, 5, 1, 1],
        'deps': ['nsubj', 'ROOT', 'compound', 'punct', 'nmod', 'dobj', 'punct']
    }),
    ("I like London and Berlin.", {
        'heads': [1, 1, 1, 2, 2, 1],
        'deps': ['nsubj', 'ROOT', 'dobj', 'cc', 'conj', 'punct']
    })
]

# give exmaples of heads and dep label - i.e. verb is word at index 0, DEP clearly defined
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=10):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the parser to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'parser' not in nlp.pipe_names:
        parser = nlp.create_pipe('parser')
        nlp.add_pipe(parser, first=True)
    # otherwise, get it, so we can add labels to it
    else:
        parser = nlp.get_pipe('parser')

    # add labels to the parser
    for _, annotations in TRAIN_DATA:
        for dep in annotations.get('deps', []):
            parser.add_label(dep)

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'parser']
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, losses=losses)
            print(losses)

    # test the trained model
    test_text = "I like securities."
    doc = nlp(test_text)
    print('Dependencies', [(t.text, t.dep_, t.head.text) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print('Dependencies', [(t.text, t.dep_, t.head.text) for t in doc])


if __name__ == '__main__':
    plac.call(main)

    # expected result:
    # [
    #   ('I', 'nsubj', 'like'),
    #   ('like', 'ROOT', 'like'),
    #   ('securities', 'dobj', 'like'),
    #   ('.', 'punct', 'like')
    # ]

"""Above rather vanilla, below follows same style as POS and NER, but adding custom semantics.
WHY? Train parsers to understand new semantic relationships or DEP among words.
Particularly interesting as able to model own DEP FOR USE-CASE; BUT caution it may not alreays output CORRECT DEP.
"""
    

# run the next code as a seperate file

#!/usr/bin/env python
# coding: utf-8
"""Using the parser to recognise your own semantics
spaCy's parser component can be used to trained to predict any type of tree
structure over your input text. You can also predict trees over whole documents
or chat logs, with connections between the sentence-roots used to annotate
discourse structure. In this example, we'll build a message parser for a common
"chat intent": finding local businesses. Our message semantics will have the
following types of relations: ROOT, PLACE, QUALITY, ATTRIBUTE, TIME, LOCATION.
"show me the best hotel in berlin"
('show', 'ROOT', 'show')
('best', 'QUALITY', 'hotel') --> hotel with QUALITY best
('hotel', 'PLACE', 'show') --> show PLACE hotel
('berlin', 'LOCATION', 'hotel') --> hotel with LOCATION berlin
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
import spacy
from pathlib import Path


# training data: texts, heads and dependency labels
# for no relation, we simply chose an arbitrary dependency label, e.g. '-'
TRAIN_DATA = [
    ("find a cafe with great wifi", {
        'heads': [0, 2, 0, 5, 5, 2],  # index of token head
        'deps': ['ROOT', '-', 'PLACE', '-', 'QUALITY', 'ATTRIBUTE']
    }),
    ("find a hotel near the beach", {
        'heads': [0, 2, 0, 5, 5, 2],
        'deps': ['ROOT', '-', 'PLACE', 'QUALITY', '-', 'ATTRIBUTE']
    }),
    ("find me the closest gym that's open late", {
        'heads': [0, 0, 4, 4, 0, 6, 4, 6, 6],
        'deps': ['ROOT', '-', '-', 'QUALITY', 'PLACE', '-', '-', 'ATTRIBUTE', 'TIME']
    }),
    ("show me the cheapest store that sells flowers", {
        'heads': [0, 0, 4, 4, 0, 4, 4, 4],  # attach "flowers" to store!
        'deps': ['ROOT', '-', '-', 'QUALITY', 'PLACE', '-', '-', 'PRODUCT']
    }),
    ("find a nice restaurant in london", {
        'heads': [0, 3, 3, 0, 3, 3],
        'deps': ['ROOT', '-', 'QUALITY', 'PLACE', '-', 'LOCATION']
    }),
    ("show me the coolest hostel in berlin", {
        'heads': [0, 0, 4, 4, 0, 4, 4],
        'deps': ['ROOT', '-', '-', 'QUALITY', 'PLACE', '-', 'LOCATION']
    }),
    ("find a good italian restaurant near work", {
        'heads': [0, 4, 4, 4, 0, 4, 5],
        'deps': ['ROOT', '-', 'QUALITY', 'ATTRIBUTE', 'PLACE', 'ATTRIBUTE', 'LOCATION']
    })
]

# Worthwhile to check training data, those new DEP shows qualities in examples
# These feedbacks are vital in building custom semantics information graph

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=5):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # We'll use the built-in dependency parser class, but we want to create a
    # fresh instance – just in case.
    if 'parser' in nlp.pipe_names:
        nlp.remove_pipe('parser')
    parser = nlp.create_pipe('parser')
    nlp.add_pipe(parser, first=True)

    for text, annotations in TRAIN_DATA:
        for dep in annotations.get('deps', []):
            parser.add_label(dep)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'parser']
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, losses=losses)
            print(losses)

    # test the trained model
    test_model(nlp)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        test_model(nlp2)


def test_model(nlp):
    texts = ["find a hotel with good wifi",
             "find me the cheapest gym near work",
             "show me the best hotel in berlin"]
    docs = nlp.pipe(texts)
    for doc in docs:
        print(doc.text)
        print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != '-'])


if __name__ == '__main__':
    plac.call(main)

    # Expected output:
    # find a hotel with good wifi
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('hotel', 'PLACE', 'find'),
    #   ('good', 'QUALITY', 'wifi'),
    #   ('wifi', 'ATTRIBUTE', 'hotel')
    # ]
    # find me the cheapest gym near work
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('cheapest', 'QUALITY', 'gym'),
    #   ('gym', 'PLACE', 'find')
    #   ('work', 'LOCATION', 'near')
    # ]
    # show me the best hotel in berlin
    # [
    #   ('show', 'ROOT', 'show'),
    #   ('best', 'QUALITY', 'hotel'),
    #   ('hotel', 'PLACE', 'show'),
    #   ('berlin', 'LOCATION', 'hotel')
    # ]
```

> Example illustrate real power of spaCy in creating custom models, both retrain model with domain ken and traing completely new DEP

- [Dependency Tree with spaCy](https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy)
- [Parsing English in 500 Lines](https://explosion.ai/blog/parsing-english-in-python)


## Topic Models

- **Previous dealt with CompLing algo and SpaCy, how to use them to ANNOTATE data and decipher sentence structure, finer details of text**
- BUT BIG PICTURE and theme! 
- Definition
> Probabilistic model having info on topics in the text
> ​    1. Topic can be theme, or underlying ideas EX corpus of news topically weather, politics, sports etc
> ​    2. Useful to Represent documents as topic DISTRIBUTIONS ! (instead of BOW or TF-IDF)
> ​    3. Cluster in topics, further zoom in one topic to decipher deeper topics/themes !!
> ​    4. Chronological / time-stamped Topic variation reveals info (DYNAMIC TOPIC MODELING)
> ​    5. NOTE TOPIC ~ Distribution(Words) NOT labelled or titled (e.g. weather is a collection of sun, temperature, storm, forecast) 
> ​    6. Human assign topic from Distribution
> ​    7. Theoretical Papers [Blei LDA](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf), [Edwin Chen](http://blog.echen.met/2011/08/22/introduction-to-latent-dirichlet-allocation/), [Blei Probabilistic Topic Models](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf)

#### Topic Models in GENSIM

- Arguably most popular TM due to many algos
    - [Topic Modelling](https://github.com/bhargavvader/personal/blob/master/notebooks/text_analysis_tutorial/topic_modelling.ipynb)
- Hierarchical Dirichlet Process **non-parametric** no hyperparam of topic-number
    - [NIPS](https://nips.cc/) and [Sharing Clusters Among Related Groups: Hierarchical Dirichlet Processes](http://papers.nips.cc/paper/2698-sharing-clusters-among-related-groups-hierarchical-dirichlet-processes.pdf)
- DTM - time-stamped evolution of topics 
    - Unlikely see underlying topic change but prominence and replacement
- GENSIM [notebook](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/ldaseqmodel.ipynb)

#### Topic Model in SKL
- Fast LDA and NMF (**NonNegative Matrix Factorization**)
- Differing to GENSIM
    1. Perplexity bounds are not expected to agree exactly here as bound is computed differently, how topics CONVERGE in TM algos
    2. SKL uses CYTHON in making numerical 6th decimal point differences
- [NMF](http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization) is LinAlg reconstructing a Single Matrix V into W and H, used to identify topics as they best represent original V - document matrix having info on words in docs
- Positive-Semi-Definite is inherent property in audio / text processing - insolvable in closed-form but numerically approx, by DISTANCE NORM Euclidean Norm2 often, and [Kullback-Leibler Divergence](https://projecteuclid.org/euclid.aoms/1177729694)
- NMF used for DimReduction, source separation, topic extraction, etc - this example uses generalised KL divergence, equivalent to [Probabilistic Latent Sementic Indexing PLSI](https://arxiv.org/ftp/arxiv/papers/1301/1301.6705.pdf)
- SLK consistent pipelien **fit, transform, predict** DECOMPOSITION ONLY USE FIT then extract components

```python
# we need to first set up the text and corpus as it was done in section 3.3
# this refers to the code set-up in the Chapter 3

from gensim.models import LdaModel

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
ldamodel.show_topics()

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
hdpmodel.show_topics()




from sklearn.decomposition import NMF, LatentDirichletAllocation
nmf = NMF(n_components=no_topic).fit(tfidf_corpus)
lda = LatentDirichletAllocation(n_topics=no_topics).fit(tf_corpus)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

display_topics(nmf, tfidf_feature_names, no_top_words)
```

## Advanced Topic Model
- Previous Preprocessing (Language Model and Vectorising-Transformation) geared more towards generating TM than other forms of Text Analysis Algo
- E.g. Lemmatisation instead of Stemming especially useful in TM as lemmatised words tend to be more legible than stemming.
- Similary bi-grams or tri-grams as part of CORPUS before applying TM give further legbility
- TM ultimately is human-understanding, unlike Clustering only higher accuracy
- Any preprocessing customised in pipeline conducive to that goal is preferred
- Multiple Runs of TM may required before any meaningful result, e.g. adding new stop words after viewing first TM
- EX removing lemmatised SAY
    ```python
    my_stop = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying']
    for stopwrod in my_stop:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True
    ```
    > For every word to add as stop, change `is_stop` attr for that `lexeme` class, which are case-insensitive, so ignorable
- A more common way to remove stop is to put all in list and remove from Corpus (e.g. in NLTK `from nltk.corpus import stopwords; stopword_list = stopwords.words("english")`
- Another way is GENSIM `Dictionary` class
    ```python
    filter_n_most_frequent(remove_n)
    from gensim.corpora import Dictionary
    corpus = [[ 'mama', 'mela', 'maso'], ['ema', 'ma', 'mama']]
    dct = Dictionary(corpus)
    dct.filter_n_most_frequent(2)
    ```
    > this process of TM, often manually inspecting and change as need is common in almost all ML or DS projects, in text, the extra is human interpretable nature of results

#### HyperParam in TM
- GENSIM
    1. `chunksize` controls # doc processed at once in training algo - speed for RAM fit
    2. `passes` controls how often train model on entire corpus or **epochs**
    3. `iterations` controls freq repeating a loop over each doc, often higher
- [LdaModel](https://radimrehurek.com/gensim/models/ldamodel.html) and [LDA in SKL](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.Laten-DirichletAllocation.html)
    1. **Alpha** repr doc-topic density, higher the more topics 
    2. **Beta** repr topic-word density
    3. **Numer of topics** 
- Logging is useful to monitor during training (GENSIM)
    ```python
    import logging
    logging.basicConfig(filename='logfile.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # document - topic proportions
    ldamodel[corpus[0]] 
    
    # printing first topic
    
    ldamodel.show_topics()[1]
    
    texts = [['bank','river','shore','water'],
            ['river','water','flow','fast','tree'],
            ['bank','water','fall','flow'],
            ['bank','bank','water','rain','river'],
            ['river','water','mud','tree'],
            ['money','transaction','bank','finance'],
            ['bank','borrow','money'], 
            ['bank','finance'],
            ['finance','money','sell','bank'],
            ['borrow','sell'],
            ['bank','loan','sell']]
    
    model.get_term_topics('water')
    model.get_term_topics('finance')
    
    bow_water = ['bank','water','bank']
    bow_finance = ['bank','finance','bank']
    bow = model.id2word.doc2bow(bow_water) # convert to bag of words format first
    doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)
    ```

- GENSIM [FAQ](https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ) and [Chris Tufts blog](https://miningthedetails.com/blog/python/lda/GensimLDA/)

#### Exploring Documents after Satisfactory TM runs
- [Top Methods](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_methods.ipynb)
> As shown, based on context, most likely topics associated with a word can vary, differing from `get_term_topics` where it is a STATIC Topic Distribution
> ​    1. NOTE GENSIM implementation of LDA uses **VARIATIONAL BAYES SAMPLING**, a `word_type` in doc is onlly given one Topic Distribution. E.g. `the bank by the river bank` is likely to be assigned to topic_0 and each of bank word instances has the same distribution
> ​    2. These 2 methods ensemble to infer further info from using TM - topic distribution means able to use info to do some visualisation - colour all words in doc based on which topic belonging to, or usng distance metrics to infer how close or far pairs of topics are
- [Distance Metric](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/distance_metrics.ipynb)
- [SKL Implementation](https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847d)

#### Topic Coherence and Evaluation
- Previous more Qualitative measures 

    > Topic Coherence [overview](https://rare-technologies.com/what-is-topic-coherence/) and [Exploring Space of Topic Coherence](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)
- In essence, TC is **quantative measure** of TM, making possible comparing two models **RIGHT, OPTIMAL NUMBER OF TOPICS** is the goal
- Before Topic Coherence, [perplexity](http://qpleple.com/perplexity-to-evalutate-topic-models/) used to measure model fit
- Resources
    1. [Coherence Model Pipeline](https://radimrehurek.com/gensim/models/coherencemodel.html)
    2. [News Classification with GENSIM](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim_news_classification.ipynb)
    3. [TC on Movies Dataset](https://github.com/Rare-Technologies/gensim/blob/develop/docs/notebooks/topic_coherence-movies.ipynb)
    4. [TC Intro](https://github.com/Rare-Technologies/gensim/blob/develop/docs/notebooks/topic_coherence_tutorial.ipynb)
    5. [TC Use Cases](https://gist.github.com/dsquareindia/ac9d3bf57579d02302f9655db8dfdd55)
    6. [TC Model Selection](https://github.com/Rare-Technologies/gensim/blob/develop/docs/notebooks/topic_coherence_model_selection.ipynb)

```python
# coherence models

lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=texts, dictionary=dictionary, window_size=10)
hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=texts, dictionary=dictionary, window_size=10)
lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary, window_size=10)

# train two models, one poorly trained (1 pass), and one trained with more passes (50 passes)

print(goodcm.get_coherence())
print(badcm.get_coherence())


c_v = []
for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary,          coherence='c_v')
        c_v.append(cm.get_coherence())
```

#### Visualising TM
- TM better understood qualitatively on textual data - visual is best ways to check
- `pyLDAvis` agnostic to model trained - beyond GENSIM or LDA - only require topic-term distributions and document-topic distributions plus basic info on corpus trained on
    ```python
    import pyLDAvis.gensim
    pyLDAvis.gensim.prepare(model, corpus, dictionary)
    ```
    1. Model is palceholder for trained LDA model for example
    2. [Full notebook](http://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb)

- Visualising Live Training Model (coherence, perplexity, etc)
    - [visdom server](https://github.com/facebookresearch/visdom)
    - [GENSIM Setup](https://github.com/parulsethi/gensim/blob/tensorboard_logs/docs/notebooks/Training_visualisations.ipynb)
    - Further viewed as CLUSTER by [T-SNE](https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html)
    - Also Clustering via [**WORD2VEC** ](https://github.com/Rare-Technologies/gensim/blob/develop/docs/notebooks/Tensorboard_visualisations.ipynb)
    - [Dendrograms](https://github.com/Rare-Technologies/gensim/blob/develop/docs/notebooks/Topic_dendrogram.ipynb)
- Visual Resources
    1. [Visualising Trend](https://de.dariah.eu/tatom/visualizing_trends.html)
    2. [Visualizing Topic Share](https://de.ariah.eu/tatom/topic_model_visualizaiton.html)
    3. [Blei Visual](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM12/paper/viewFile/4645/5021)

## Clustering and Classifying

#### Clustering
- RECAP
    > so far processing text or corpus via POS, NER, what kind of words present; in TM to seek theme hidden;
    > ​    1. TM could be used to cluster articles, BUT it is NOT its purpose!
    > ​    2. E.g. after performing TM, a doc can be made of 30% topic 1, 30% topic 2, etc, hence no way to cluster
- Datapoint as documents or words 
- EXTRA CAUTION of Text: high number of dimension in text vector !! - entire vocab or corpus (Best effort via of DimReduction via SVD, LDA, LSI, etc)
- Pipeline: rid of stop, lemmatise, vectorise
- [Example](https://github.com/bhargavvader/personal/blob/master/notebooks/clustering_classing.ipynb)
- [Doc2Vec Clustering](https://towardsdatascience.com/automatic-topic-clustering-using-doc2vec-e1cea88449c)

```python
# using scikit-learn

from sklearn.datasets import fetch_20newsgroups


categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

labels = dataset.target
true_k = np.unique(labels).shape[0]
data = dataset.data  

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=True)

X = vectorizer.fit_transform(data)


from sklearn.decomposition import PCA


newsgroups_train = fetch_20newsgroups(subset='train', 
                                      categories=['alt.atheism', 'sci.space'])
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])        
X_visualise = pipeline.fit_transform(newsgroups_train.data).todense()

pca = PCA(n_components=2).fit(X_visualise)
data2D = pca.transform(X_visualise)
plt.scatter(data2D[:,0], data2D[:,1], c=newsgroups_train.target)


n_components = 5
svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)


Minibatch = True
if minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km.fit(X)

original_space_centroids = svd.inverse_transform(km.cluster_centers_) 

order_centroids = original_space_centroids.argsort()[:, ::-1]

# [The above bit of code is necessary because of our LSI transformation]

terms = vectorizer.get_feature_names()

for i in range(true_k):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(X)

from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) 
fig, ax = plt.subplots(figsize=(10, 15)) # set size
ax = dendrogram(linkage_matrix, orientation="right")
```

#### Classifying

```python
# classification

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, labels)

from sklearn.svm import SVC
svm = SVC()
svm.fit(X, labels)
```


## Similarity Queries and Summarisation
- Vectorised Text opens door to simiarlity or distance

#### Similarity Metrics
- [notebook](https://github.com/Rare-Technologies/gensim/blob/develop/docs/notebooks/distance.metrics.ipynb)
#### Similarity Queries
- extract out most similar for an input query - simply index each of doc then search for lowest distance returned between corpus and query, and return the docu with lowest distance

```python
# make sure to have appropriate gensim installations and imports done

texts = [['bank','river','shore','water'],
        ['river','water','flow','fast','tree'],
        ['bank','water','fall','flow'],
        ['bank','bank','water','rain','river'],
        ['river','water','mud','tree'],
        ['money','transaction','bank','finance'],
        ['bank','borrow','money'], 
        ['bank','finance'],
        ['finance','money','sell','bank'],
        ['borrow','sell'],
        ['bank', 'loan', 'sell']

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus)
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2)

model.show_topics()

doc_water = ['river', 'water', 'shore']
doc_finance = ['finance', 'money', 'sell']
doc_bank = ['finance', 'bank', 'tree', 'water']

bow_water = model.id2word.doc2bow(doc_water)   
bow_finance = model.id2word.doc2bow(doc_finance)   
bow_bank = model.id2word.doc2bow(doc_bank)   

lda_bow_water = model[bow_water]
lda_bow_finance = model[bow_finance]
lda_bow_bank = model[bow_bank]

tfidf_bow_water = tfidf[bow_water]
tfidf_bow_finance = tfidf[bow_finance]
tfidf_bow_bank = tfidf[bow_bank]

from gensim.matutils import kullback_leibler, jaccard, hellinger

hellinger(lda_bow_water, lda_bow_finance)
hellinger(lda_bow_finance, lda_bow_bank)
hellinger(lda_bow_bank, lda_bow_water)

hellinger(lda_bow_finance, lda_bow_water)
kullback_leibler(lda_bow_water, lda_bow_bank)
kullback_leibler(lda_bow_bank, lda_bow_water)


jaccard(bow_water, bow_bank)
jaccard(doc_water, doc_bank)
jaccard(['word'], ['word'])

def make_topics_bow(topic):
    # takes the string returned by model.show_topics()
    # split on strings to get topics and the probabilities
    topic = topic.split('+')
    # list to store topic bows
    topic_bow = []
    for word in topic:
        # split probability and word
        prob, word = word.split('*')
        # get rid of spaces
        word = word.replace(" ","")
        # convert to word_type
        word = model.id2word.doc2bow([word])[0][0]
        topic_bow.append((word, float(prob)))
    return topic_bow


topic_water, topic_finance = model.show_topics()
finance_distribution = make_topics_bow(topic_finance[1])
water_distribution = make_topics_bow(topic_water[1])

hellinger(water_distribution, finance_distribution)

from gensim import similarities

index = similarities.MatrixSimilarity(model[corpus])
sims = index[lda_bow_finance]
print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])

for doc_id, similarity in sims:
    print texts[doc_id], similarity

from gensim.summarization import summarize
print (summarize(text))

print (summarize(text, word_count=50))

from gensim.summarization import keywords

print (keywords(text))

from gensim.summarization import mz_keywords
mz_keywords(text,scores=True,weighted=False,threshold=1.0)
```

- Resources
    1. [Wiki Query](https://radimrehurek.com/topic_modeling_tutorial/3%20%20Indexing%20and%20Retrieval.html)
    2. [Simserver Tutorial](https://radimrehurek.com/gensim/simserver.html)
    3. [GENIM SIMSERVER CODE](https://github.com/RaRe-Technologies/gensim-simserver)

#### Summarising Text
- GENSIM algo **TextRank** from [Mihalcea](http://webeecs.umich.edu/~michalcea/papers/mihalcea.emnlp04.pdf)
- Improved [BM25 Ranking Function](https://arxiv.org/pdf/1602.03606.pdf)
- [Montemurro and Zanettes MZ entropy-based keyword extraction algo](https://arxiv.org/abs/0907.1558)


## Word2Vec, Doc2Vec in GENSIM
- **Word Embedding** magic W2V is how it **manages to capture semantic repr of wrods in a vector** based on many papers
- **V(King) - V(Man) + V(Woman) approx V(Queen) or V(Vietname) + V(Capital) approx V(Hanoi)**
- Concept
    > W2V sliding window size attempting to ID Cond-Proba of observing output word based on adjacent ones EX 
    >   - Two Methods for W2V training
    >       1. Continuous BOW (CBOW)
    >       2. Skip Gram - [Word2Vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model)\
    >   - [The amazing power of word vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors)
    >   - [Resources Page](http://mccormickml.com/2016/04/27/word2vec-resources/)
    >     while it remians most POPULAR word vectoriser, not first time attempted not last - others to follow

#### W2V with GENSIM
- [Code history](https://rare-technologies.com/deep-learning-with-word2vec-and-gensim)
- [Online Interactive Tutorial](https://rare-technologies.com/word2vec-tutorial)
- #### Importancy of `word2vec` class and `KeyedVector` which tuning relies heavily on
    > List of Params for `word2vec.Word2Vec`
    > ​    1. SG defines algo default=0 CBOW used or =1 Skip-gram
    > ​    2. SIZE dimensionality of feature vectors
    > ​    3. WINDOW max distance entre current-predicted word within a sentence
    > ​    4. ALPHA initial learning rate (linearly drop to `min_alpha`)
    > ​    5. SEED randNumGenerator, initial vectors for each word seeded with hash of concatenation of word + str(seed), NOTE for fully deterministically reproducible run, must also LIMIT the model to SINGLE WORKER THREAD, eliminating ordering jitter from OS thread scheduling
    > ​    6. MIN_COUNT ignore all words with a total freq lower
    > ​    7. MAX_VOCAB_SIZE lmit RAM during vocab building; if more unique, prune infrequent ones. Every 10m word types need about 1 GB of RAM (None for no limit as default)
    > ​    8. SAMPLE threshold for configuring which higer-freq words randomly downsampled; default 1e-3 useful (0, 1e-5)
    > ​    9. WORKERS use threads to train (faster with multicore)
    > ​    10. HS if 1, hierarchical softmax used else 0 negative is non-zero, negative sampling used
    > ​    11. NEGATIVE if > 0, negative smaple
    > ​    12. CBOW_MEAN if 0, use sum of context word vectors, 1 for mean
    > ​    13. HASHFXN hash func use to randomly init weights, for rised training reproducibility - default is Python\s rudimentary hash func
    > ​    14. ITER num of iterations or epochs over corpus default 5
    > ​    15. TRIM_RULE vocab trimming rule discard if word count < min_cuount (If none, min_count used, or callable accpeting params like word, count and min_count, returns either utils.RULE_DISCARD, UTILS.RULE_KEEP OR UTILS.RULE_DEFAULT) NOTE if given, only used to prune during build_vocab and not stored as part of model
    > ​    16. SORTED_VOCAB if 1 default sort desc before assigning index
    > ​    17. BATCH_WORDS target size in words passed to worker threas default 10k
- [Notebook](https://github.com/bhargavvader/personal/blob/master/notebooks/text_analysis/word2vec.ipynb)
- [GENSIM TUtorial](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/online_w2v_tutorial.ipynb)
- Training on generic corpus preferable - [Text8 from Wiki](http://mattmahoney.net/dc/text/data.html)
- GENSIM allows **similar API to download models using other word EMBEDDINGS
- Equipped to train, load models, use word embeddings to conduct experiments 

```python
# be sure to make appropriate imports and installations

from gensim.models import word2vec

sentences = word2vec.Text8Corpus('text8') 
model = word2vec.Word2Vec(sentences, size=200, hs=1)

print(model)
model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)[0]

model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])

model.wv['computer']model.save("text8_model")

model.save("text8_model")
model = word2vec.Word2Vec.load("text8_model")

model.wv.doesnt_match("breakfast cereal dinner lunch".split())

model.wv.similarity('woman', 'man')

model.wv.similarity('woman', 'cereal')
 
model.wv.distance('man', 'woman')


word_vectors = model.wv
del model

model.wv.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
model.wv.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))

from gensim.models import KeyedVectors
# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

```



### Doc2Vec
- Extending Word2Vec with another vector **paragraph ID**
    1. [Distributed Representations of Senatences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
    2. [A gentle Introduction to Doc2Vec](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
- One major diff about GENSIM is not expecting a simple corpus as intpu - algo expects TAGS or LABELS as part of input

```python
gensim.models.doc2vec.LabeledSentence
# alternatively
gensim.models.doc2vec.TaggedDocument
sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])
# in case of error, try
sentence = LabeledSentence(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])
```
    > Here`sentence` an example of what input resembles
- [Tutorial Notebook based on LEE Corpus](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb)

```python
# LEE corpus

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

def read_corpus(file_name, tokens_only=False):
    with smart_open.smart_open(file_name) as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=100)
model.build_vocab(train_corpus) 
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

models = [
    # PV-DBOW 
    Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8, min_count=10, epochs=50),
    
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=8, min_count=10, epochs =50),
]

models[0].build_vocab(documents)
models[1].reset_from(models[0])

for model in models:
   model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
new_model = ConcatenatedDoc2Vec((models[0], models[1]))

inferred_vector = model.infer_vector(train_corpus[0].words)
sims = model.docvecs.most_similar([inferred_vector])
print(sims)
```
    > In practice, no need to test for most similar vectors on training set - this is to illustrate
    1. Note list of doc most similar to doc 0 ID 0 shows up first - more interesting to check 48th or 255th doc

> **Context captured perfectly by Doc2Vec, simply searched up the most similar doc - imagine the power it brings if used in tandem with clustering and classifying doc ! Instead of TF-IDF or TM as previously presented** !!

> **Such is Vectorisation with SEMANTIC understanding both words and documents**

### Other Word Embeddings
- GENSIM wraps most of popular methods 

    > WORDRANK, VAREMBED, FASTTEXT, POINCARE EMBEDDINGS
- Neat script to use **GloVe embeddings** useful in comparing between diff kinds of embeddings
- `KeyedVectors` class is base to use all word embeddgins
- Key to note is RUN `word_vectors = model.wv` AFTER done training model
- Also, continue using `word_vectors` for all tasks - for most similar words, most dissimilar and running tests for word embeddings - [source code of KeyedVectors.py](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.ipynb)

#### GloVe
- Training done on aggregated global word-word co-occurrence stats from a corpus - like Word2Vec, using context to decipher and create word representations
- Developed by NLPL Stanford and [paper](https://nlp.stanford.edu/pubs/glove.pdf) worth reading as it illustrates some of the pitfalls of LSA and Word2Vec 
- Many implementations and even in Python system - not training here but using (training need to tweek [glove_python](https://github.com/maciejkula/glove-python) or just [glove](https://github.com/JonathanRaiman/glove) or look at [source](https://github.com/stanfordnlp/GloVe)
- GENSIM
    1. download or train GloVe vectors - save - convert format to Word2Vec for futher usg in GENSIM API
    2. Download [page](https://nlp.stanford.edu/projects/glove)

```python
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
```

#### FastText
- Dev by Facebook AI, fast and efficient due to morphological details learning
- Unique in deriving word vectors for unknown owrds from morphological char of words, creating word vector for unseen
- Intriguing in some langugage, English e.g. 'ly' `embedding(strang) - embedding(strangely) ~= embedding(charming) - embedding(charmingly)`
- Performs better for structure or syntax, while Word2Vec for semantic tasks
- [Notebook](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Word2Vc_FastText_Comparison.ipynb) and [Doc](https://radimrehurek.com/gensim/models/fastext.html#module-gensim.models.fasttext)
- Possible to use C++ via [wrapper](https://radimrehurek.com/gensim/models/wrappers/fasttext.html)
- [Notebook1](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb)

```python
from gensim.models.wrappers.fasttext import FastText

# Set FastText home to the path to the FastText executable
ft_home = '/home/bhargav/Gensim/fastText/fasttext'
# train the model
model_wrapper = FastText.train(ft_home, train_file)

print('dog' in model.wv.vocab)
print('dogs' in model.wv.vocab)

print('dog' in model)
print('dogs' in model)
```

#### WordRank
- Embedding as Ranking - similar to GloVe in using global co-occurences of words to generate
- [code](https://bitbucket.org/shihaoji/wordrank) and [github](https://github.com/shihaoji/wordrank)
- GENSIM API - beware of `dump_period` and `iter` param needed to be sync as it dumps file with start of next iteration [Tutorial](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/WordRank_wrapper_quick-start.ipynb)
- **CAVEAT** window size of 15 performed with optimum results, and 100 epochs is better than 500, quite long. 
- GOOD COMPARISON betwee FastText, word2vec and WorkRank [blog](https://rare-technologies.com/wordrank-embedding-crowned-is-most-similar-to-king-not-word2vecs-canute) and [Notebook](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Wordrank_comparisons.ipynb)

#### Varembed
- Like FastText it takes morphological info to generate word vectors
- Similar to GloVe, cannot update model with new words and need to train a new model [code](https://github.com/rguthrie3/MorphologicalPriorsForWrodEmeddings)

```python
from gensim.models.wrappers import varembed
varembed_vectors = '../../gensim/test/test_data/varembed_leecorpus_vectors.pkl'
model = varembed.VarEmbed.load_varembed_format(vectors=varembed_vectors)


morfessors = '../../gensim/test/test_data/varembed_leecorpus_morfessor.bin'
model = varembed.VarEmbed.load_varembed_format(vectors=varembed_vectors, morfessor_model=morfessors)
```

#### Poincare
- Also dev by FB - using graphical repr of words to decipher relationship between words to generate vector
- Also captures hiearchical info computed by hyperbolic space not traditioanl Norm2 allowing for hierarchy info
- [Poincaré Embeddings for Learning Hiearchical Represenstaions](https://arxiv.org/pdf/1705.08039.pdf)

```python
import os

poincare_directory = os.path.join(os.getcwd(), 'docs', 'notebooks', 'poincare')
data_directory = os.path.join(poincare_directory, 'data')
wordnet_mammal_file = os.path.join(data_directory, 'wordnet_mammal_hypernyms.tsv')

# Training process
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors, PoincareRelations
relations = PoincareRelations(file_path=wordnet_mammal_file, delimiter='\t')
model = PoincareModel(train_data=relations, size=2, burn_in=0)
model.train(epochs=1, print_every=500)

# Also use own iterable of relations to train model
# each relation is just a pair of nodes
# GENSIM also has pre-trained models as follows

models_directory = os.path.join(poincare_directory, 'models')
test_model_path = os.path.join(models_directory, 'gensim_model_batch_size_10_burn_in_0_epochs_50_neg_20_dim_50')
model = PoincareModel.load(test_model_path)
```
- [Training](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Poincare%20Tutorial.ipynb) and [Blog](https://rare-technologies.com/implementing-poincare-embeddings)

## Deep Learning for Text
### Generating Text
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs)
- [Unreasonable effectiveness of NN](http://karpathy.github.io/2015/05/21/rnn-effectiveness)
- [Notebook](https://github.com/kirit93/Personal/blob/master/text_generation_keras/text_generation.ipynb)

```python
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np

# Any text source as input based on what kind of data to generate
# CREATEIVE! HERE - RNN to write poetry if enough data
# Need to generate MAPPING of all distinct characters inf book (LSTM is char-level model)

filename    = 'data/source_data.txt'
data        = open(filename).read()
data        = data.lower()
# Find all the unique characters
chars       = sorted(list(set(data)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
ix_to_char  = dict((i, c) for i, c in enumerate(chars))
vocab_size  = len(chars)

# The 2 dicts pass char to model and in generating
# RNN accepts seq of char as input and ouput such similar seq - to break up into seq
seq_length = 100
list_X = [ ]
list_Y = [ ]
for i in range(0, len(chars) - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	list_X.append([char_to_int[char] for char in seq_in])
	list_Y.append(char_to_int[seq_out])
n_patterns = len(dataX)

X  = np.reshape(list_X, (n_patterns, seq_length, 1)) 
# Encode output as one-hot vector
Y  = np_utils.to_categorical(list_Y)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Dropout to control overfitting 

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

# callback save weights to fiile at whenever improvement

# OR transfer learning
filename = "weights.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# generate text start text randomly
start   = np.random.randint(0, len(X) - 1)
pattern = np.ravel(X[start]).tolist()

output = []
for i in range(250):
    x           = np.reshape(pattern, (1, len(pattern), 1))
    x           = x / float(vocab_size)
    prediction  = model.predict(x, verbose = 0)
    index       = np.argmax(prediction)
    result      = index
    output.append(result)
    pattern.append(index)
    pattern = pattern[1 : len(pattern)]

print ("\"", ''.join([ix_to_char[value] for value in output]), "\"")

# IDEA: based on X input, choose highest porba for next char using argmax, convert that index to a char, append it to output list, loops = iterations in output
```

- Resources
    1. [NLP Best Practice](http://ruder.io/deep-learning-nlp-best-practices/index.html#bestpractices)
    2. [Deep Learning and Representation](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations)
    3. [Best of 2017 for NLP and DL](https://tryolabs.com/blog/2017/12/12/deep-learning-for-nlp-advancements-and-trends-in-2017)

## Keras and SpaCy for DL
- [Keras Sequential Model](https://keras.io/getting-started/sequential-model-guide)
- [Keras CNN LSTM](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py)
- [Pre-trained word embeddings](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)

#### Keras and SpaCy
- [Keras Submodule Text Preprocess](https://keras.io/preprocessing/text)
- **KERAS KEN**
    1. [About Keras Models: explains various kinds of NN in Keras](https://keras.io/models/about-keras-models)
    2. [About Keras Layers](https://keras.io/layers/about-keras-layers)
    3. [Core Layers](https://keras.io/layers/core)
    4. [Keras Datasets](https://keras.io/datasets)
    5. [LSTM](https://keras.io/layers/recurrent/#lstm)
    6. [CNN](https://keras.io/layers/convolutional)
> SpaCy `TextCategorizer` trains similar to other components as POS and NER, also integrating wiht other word embeddings such as GENSIM Word2Vec or GloVe, plus plug-in to Keras model; **Combine SpaCy and Keras allows powerful classification machien**

#### Classification with Keras
- **Small dataset such as IMDB might get better result using simply BOW + SVM**

```python

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
```

> Notes
> ​    1. Not using text preprocess moduels as IMDB dataset already cleaned
> ​    2. LSTM for classification, a variant of RNN
> ​    3. LSTM is mere Layer inside `Sequential` model

```python
max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
```
> `max_features` refers to top words wishe to use limited to 20k words; similar to ridding of least used words; `maxlen` used for fix length as NN accpets a FIED LEN input; `batch_size` used later to specify batches trained

```python
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
```
> Setup Seq-model, stacked on layers of word-embeddings (20k features), dropped down to 128 **Option to use other embedders** - LSTM 128 number of Dimensions

```python
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))


score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
```
- CNN need a few more params to tune

```python
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2
```
> Above params affects training heavily and are empirically derived after experiemtns

```python
print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))
```
> 7 payers, Pooling layer to progressively reduce spatial size to reduce params hence controlling overfitting; 

```python
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
```
> Below use pretrained word embeddings in classifier to improve results

```python
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# using preceding var/arg to load word embeddgins
print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# Simple loop through files 
print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Make sure set training argument to false so to use word vectors as is
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

# Layers stacked differently with x var holding each layer
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))
```
> Here used different measure for calcu loss; above illustrates basic LSTM, a CNN and a CNN using pretrained word embeddings
> ​    1. See progressive rise in performance of each networkds
> ​    2. Embeddings are esp. useful when not much data
> ​    3. CNN generally perform better than Sequential, those using word-embedding even better
> ​    4. Useful to train and compare with Non-NN model such as NB or SVM

#### Classification with SpaCy
- While keras works esp. well in standalone text classification, it might be useuful to use Keras plus spaCy
- 2 Ways to Text Classification in SpaCy
    1. Own NN library **THINC**
    2. Keras
- Example 1 [code](https://github.com/explosion/spaCy/blob/master/examples/deep_learning_keras.py)
    > Set up
    > ​    1. This example shows how to use an LSTM sentiment classification model trained using Keras in spaCy. spaCy splits the document into sentences, and each sentence is classified using the LSTM. The scores for the sentences are then aggregated to give the document score. 
    > ​    2. This kind of hierarchical model is quite difficult in "pure" Keras or Tensorflow, but it's very effective. The Keras example on this dataset performs quite poorly, because it cuts off the documents so that they're a fixed size. This hurts review accuracy a lot, because people often summarise their rating in the final sentence
    > ​    3. Prerequesit: spacy download en_vectors_web_lg / pip install keras==2.0.9 / Compatible with: spaCy v2.0.0+

```python
import plac
import random
import pathlib
import cytoolz
import numpy
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import thinc.extra.datasets
from spacy.compat import pickle
import spacy

class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_length=100):
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    def __init__(self, model, max_length=100):
        self._model = model
        self.max_length = max_length

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)
        
    # set up class and how to load model and embedding weights
    # INIT model, max length, instructions to predict
    # Load method returns model to use in eval
    # call gets features and pred

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            for sent, label in zip(sentences, ys):
                sent.doc.sentiment += label - 0.5
            for doc in minibatch:
                yield doc

    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y
        
# 
def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


def get_features(docs, max_length):
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs

# Below is where all heavy lifting place - NOTE lines involving spacy's pipeline of sentencizer
def train(train_texts, train_labels, dev_texts, dev_labels,
          lstm_shape, lstm_settings, lstm_optimizer, batch_size=100,
          nb_epoch=5, by_sentence=True):
    print("Loading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              nb_epoch=nb_epoch, batch_size=batch_size)
    return model

# Like previously, set up each layers and stack up
# Any Keras model would do, here bidriectional LSTM
def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'],
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
          metrics=['accuracy'])
    return model

# Eval method retunrs a score of how well model performed; checks assigned sentiment score with label of document
def get_embeddings(vocab):
    return vocab.vectors.data


def evaluate(model_dir, texts, labels, max_length=100):
    def create_pipeline(nlp):
        '''
        This could be a lambda, but named functions are easier to read in Python.
        '''
        return [nlp.tagger, nlp.parser, SentimentAnalyser.load(model_dir, nlp,
                                                               max_length=max_length)]

    nlp = spacy.load('en')
    nlp.pipeline = create_pipeline(nlp)

    correct = 0
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        correct += bool(doc.sentiment >= 0.5) == bool(labels[i])
        i += 1
    return float(correct) / i

# Using IMBD sentiment analysis datteset, this method is an API to access data
def read_data(data_dir, limit=0):
    examples = []
    for subdir, label in (('pos', 1), ('neg', 0)):
        for filename in (data_dir / subdir).iterdir():
            with filename.open() as file_:
                text = file_.read()
            examples.append((text, label))
    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return zip(*examples) # Unzips into two lists

# Annotations set up options setting various model directories, runtime, and params 
@plac.annotations(
    train_dir=("Location of training file or directory"),
    dev_dir=("Location of development file or directory"),
    model_dir=("Location of output model directory",),
    is_runtime=("Demonstrate run-time usage", "flag", "r", bool),
    nr_hidden=("Number of hidden units", "option", "H", int),
    max_length=("Maximum sentence length", "option", "L", int),
    dropout=("Dropout", "option", "d", float),
    learn_rate=("Learn rate", "option", "e", float),
    nb_epoch=("Number of training epochs", "option", "i", int),
    batch_size=("Size of minibatches for training LSTM", "option", "b", int),
    nr_examples=("Limit to N examples", "option", "n", int)
)


# Now the main functions
def main(model_dir=None, train_dir=None, dev_dir=None,
         is_runtime=False,
         nr_hidden=64, max_length=100, # Shape
         dropout=0.5, learn_rate=0.001, # General NN config
         nb_epoch=5, batch_size=100, nr_examples=-1):  # Training params
    if model_dir is not None:
        model_dir = pathlib.Path(model_dir)
    if train_dir is None or dev_dir is None:
        imdb_data = thinc.extra.datasets.imdb()
    if is_runtime:
        if dev_dir is None:
            dev_texts, dev_labels = zip(*imdb_data[1])
        else:
            dev_texts, dev_labels = read_data(dev_dir)
        acc = evaluate(model_dir, dev_texts, dev_labels, max_length=max_length)
        print(acc)
    else:
        if train_dir is None:
            train_texts, train_labels = zip(*imdb_data[0])
        else:
            print("Read data")
            train_texts, train_labels = read_data(train_dir, limit=nr_examples)
        if dev_dir is None:
            dev_texts, dev_labels = zip(*imdb_data[1])
        else:
            dev_texts, dev_labels = read_data(dev_dir, imdb_data, limit=nr_examples)
        train_labels = numpy.asarray(train_labels, dtype='int32')
        dev_labels = numpy.asarray(dev_labels, dtype='int32')
        lstm = train(train_texts, train_labels, dev_texts, dev_labels,
                     {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 1},
                     {'dropout': dropout, 'lr': learn_rate},
                     {},
                     nb_epoch=nb_epoch, batch_size=batch_size)
        weights = lstm.get_weights()
        if model_dir is not None:
            with (model_dir / 'model').open('wb') as file_:
                pickle.dump(weights[1:], file_)
            with (model_dir / 'config.json').open('wb') as file_:
                file_.write(lstm.to_json())


if __name__ == '__main__':
    plac.call(main)
```
> First few lines set up model folder and load dataset / then check print run time info, if not training is not complete proceeding to train, train and save model 
> ​    1. Running, saving using model in pipelines is huge motif behind Keras and SpaCy in such a way
> ​    2. KEY here updating `sentiment` attri for each doc (how is optional)
> ​    3. SpaCy GOOD at not removing or truncating input - as users tend to sum up review in last sentence of documents with a lot of sentiment inferred
> ​    4. HOW to USE? model adds one more attribute to doc, `doc.sentiment` caputuring 
> ​    5. Verify by loading saved model and run any document through pipelien the same way through prevous POS, NER and Dep-parsing `doc = nlp(document)`
> ​    6. `nlp` is pipeline obj of loaded model trained, `docuemnt` any unicode text wish to analyse

- Non-NN classifier
    1. proba of duc belonging to particular class
    2. simple, use `update` [code](https://spacy.io/usage/training#section-textcat) and [github](https://github.com/explosion/spacy/blob/master/examples/training/train_textcat.py)
- Example below to run at once

```python
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import thinc.extra.datasets

import spacy
from spacy.util import minibatch, compounding

# Not Keras, but SpaCy's THINC

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=20, n_texts=2000):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    textcat.add_label('POSITIVE')

    # load the IMDB dataset
    print("Loading IMDB data...")
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)
    print("Using {} examples ({} training, {} evaluation)"
          .format(n_texts, len(train_texts), len(dev_texts)))
    train_data = list(zip(train_texts,
                          [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))

    # test the trained model
    test_text = "This movie sucked"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)

# test model with eval method, calcu precision, recall F-score, last part saving trained model in output dir

def load_data(limit=0, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    train_data, _ = thinc.extra.datasets.imdb()
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{'POSITIVE': bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


if __name__ == '__main__':
    plac.call(main)
```

> Final methods similar to before in MAIN fucn; one is to load data, other to eval; return data appropriately shuffled and split, eval func calcu true neatves, TP, FN and FP to create confusion matrix

```python
test_text = "This movie disappointed me severely"
doc = nlp(test_text)
print(test_text, doc.cats)
```
> `doc.cats` gives result of classifcation, i.e. negative sentiments

> **Such is final setp - test model on sample, also see one of main pros of spaCy for DL - fits seamlessly in PIPELINE, and classifaction or sentiment score ends up being antoehr attribute of document - quite differente to Keras, whose purpose is to EITHER generating text OR to output proba-Vectors (vector in vector out); Possible to leverage this info as part of text analysis pipeline BUT spaCy does training under hood and learns attributes to doc makes easy to include info as part of any text analysis PIPELINE**

## Ideas for Project
#### Reddit Sense2Vec SpaCy
- [Code](https://github.com/explosion/sense2vec) Semantic analysis modifiable in source data and **Semantics** and web app! Visualisation

#### Twitter Mining
- [Label Dataset](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22) and [UMichigan Kaggle](https://www.kaggle.com/c/si650winter11) and [Sentiment140 dataset](http://help.sentiment140.com/for-students)

#### Chatbot
- [A Neural Conversational Model - Vinyal and Lee](https://arxiv.org/pdf/1506.05869v1.pdf)
- Production-grade API **RASA NLU** and **ChatterBot**
- RASA [JSON-data Example](https://github.com/RASAHQ/rasa_nlu/blob/master/data/examples/rasa/demo-rasa.json)
    > adding more entites and intent, model laerns more context better decipher questions - one of backend is SpaCy and SKL
    > ​    1. UnderHood, Word2Vec for intent, spaCy clean up text, SLK build models - [detail](https://medium.com/rasa-blog/do-it-yourself-nlp-for-bot-developers-2e2da2817f3f) 
    > ​    2. One of which involves being able to write own parts of bot instead of API
    > ​    3. JSON entry/data to train RASA (see elsewhere)
- [Front-end](https://core.rasa.com) and [Tutorial](https://core.rasa.com/tutorial_basics.html)
- Non-AI but Learn-Concept at [Chatbot Fundamentals](https://apps.worldwritable.com/tutorials/chatbot) and [Brobot](https://github.com/lizadaly/brobot)
- Recall the Concept of chatbot 
    1. Take Input
    2. Classify intent (question, statement, greeting)
    3. If greeting - 
    4. If Question (query simialr questions from dataset, do sentence analysis, response e.e. based on Reddit/Food or Twitter conversion)
    5. If statement/conversion (generative model)
    6. Closure




# Polyglot NER `polygplot`
- Vector word
- Why? main is language.... over 130
- e.g. transliteration
- practice Spanish NER with polyglot
- auto-detect once init 'langue'


```python
from polyglot.text import Text

txt = Text(article_uber)


```


    ---------------------------------------------------------------------------
    
    ModuleNotFoundError                       Traceback (most recent call last)
    
    <ipython-input-154-47870b8e6243> in <module>()
    ----> 1 from polyglot.text import Text
          2 
          3 txt = Text(article_uber)


    ~/anaconda3/lib/python3.6/site-packages/polyglot/text.py in <module>()
          7 
          8 from polyglot.base import Sequence, TextFile, TextFiles
    ----> 9 from polyglot.detect import Detector, Language
         10 from polyglot.decorators import cached_property
         11 from polyglot.downloader import Downloader


    ~/anaconda3/lib/python3.6/site-packages/polyglot/detect/__init__.py in <module>()
    ----> 1 from .base import Detector, Language
          2 
          3 __all__ = ['Detector', 'Language']


    ~/anaconda3/lib/python3.6/site-packages/polyglot/detect/base.py in <module>()
          9 
         10 
    ---> 11 from icu import Locale
         12 import pycld2 as cld2
         13 


    ModuleNotFoundError: No module named 'icu'


```python
# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent)
    
# Print the type of ent
print(type(ent))


# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print entities
print(entities)
```

### Spanish NER
```python
# Initialize the count variable: count
count = 0

# Iterate over all the entities
for ent in txt.entities:
    # Check whether the entity contains 'Márquez' or 'Gabo'
    if "Márquez" in ent or "Gabo" in ent:
        # Increment count
        count += 1

# Print count
print(count)

# Calculate the percentage of entities that refer to "Gabo": percentage
percentage = count / len(txt.entities)
print(percentage)
```


# SL with NLP
- Classification of **fake news**
- Use language au lieu de Features
- Creating train data from text
    1. BOW or tf-idf as **feature**

## IMDB Movie Example
- Plot **text data**
- Type of Film **MultiCAT**
- Target: Predict genre by plot
## Possible Features in Text-Classification
- Frequency (BOW or tf-idf)
- Topic (Named Entities)
- Language



```python
df_movie = pd.read_csv('Data_Folder/TxT/fake_or_real_news.csv')
df_movie.head()
df_movie.label[:5]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

y = df_movie.label

X_train, X_test, y_train, y_test = train_test_split(df_movie.text, y, test_size=0.33, random_state=53)

count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(X_train)

count_test = count_vectorizer.transform(X_test)

print(count_vectorizer.get_feature_names()[:10])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8476</td>
      <td>You Can Smell Hillary’s Fear</td>
      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10294</td>
      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>
      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3608</td>
      <td>Kerry to go to Paris in gesture of sympathy</td>
      <td>U.S. Secretary of State John F. Kerry said Mon...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10142</td>
      <td>Bernie supporters on Twitter erupt in anger ag...</td>
      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>875</td>
      <td>The Battle of New York: Why This Primary Matters</td>
      <td>It's primary day in New York and front-runners...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>






    0    FAKE
    1    FAKE
    2    REAL
    3    FAKE
    4    REAL
    Name: label, dtype: object



    ['00', '000', '0000', '00000031', '000035', '00006', '0001', '0001pt', '000ft', '000km']



```python
# similar to sparse CountVectorizer, create tf-idf vectors

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(X_train)

tfidf_test = tfidf_vectorizer.transform(X_test)

print(tfidf_vectorizer.get_feature_names()[:10])

print(tfidf_train.A[:5])

```

    ['00', '000', '0000', '00000031', '000035', '00006', '0001', '0001pt', '000ft', '000km']
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]



```python
# some inspection

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

print(count_df.head(), '\n', tfidf_df.head(), '\n')

difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference, '\n')

print(count_df.equals(tfidf_df))

```

       00  000  0000  00000031  000035  00006  0001  0001pt  000ft  000km  ...    \
    0   0    0     0         0       0      0     0       0      0      0  ...     
    1   0    0     0         0       0      0     0       0      0      0  ...     
    2   0    0     0         0       0      0     0       0      0      0  ...     
    3   0    0     0         0       0      0     0       0      0      0  ...     
    4   0    0     0         0       0      0     0       0      0      0  ...     
    
       حلب  عربي  عن  لم  ما  محاولات  من  هذا  والمرضى  ยงade  
    0    0     0   0   0   0        0   0    0        0      0  
    1    0     0   0   0   0        0   0    0        0      0  
    2    0     0   0   0   0        0   0    0        0      0  
    3    0     0   0   0   0        0   0    0        0      0  
    4    0     0   0   0   0        0   0    0        0      0  
    
    [5 rows x 56922 columns] 
         00  000  0000  00000031  000035  00006  0001  0001pt  000ft  000km  ...    \
    0  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    1  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    2  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    3  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    4  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    
       حلب  عربي   عن   لم   ما  محاولات   من  هذا  والمرضى  ยงade  
    0  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    1  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    2  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    3  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    4  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    
    [5 rows x 56922 columns] 
    
    set() 
    
    False

### Testing using Naive Bayes Classifier

- NB Model commonly used for testing NLP classificaiton problems
- basis in probability
- likelihood estimation
- Conditional Probability of token
- Not working well with float like tf-idf



```python
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

nb_classifier.fit(count_train, y_train)

pred = nb_classifier.predict(count_test)

score = metrics.accuracy_score(y_test, pred)
print(score)

cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)

```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



    0.893352462936394
    [[ 865  143]
     [  80 1003]]



```python
# NB on tf-idf
nb_classifier_tfidf = MultinomialNB()

nb_classifier_tfidf.fit(tfidf_train, y_train)

pred_tfidf = nb_classifier.predict(tfidf_test)

score_tfidf = metrics.accuracy_score(y_test, pred)
print(score_tfidf)

cm_tfidf = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm_tfidf)

```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



    0.893352462936394
    [[ 865  143]
     [  80 1003]]

### Simple NLP, Complex Problems

- Translation, grammar in languages
- Word complexity in other languages
- Sentiment shift
- Semantics
- Custom in gender, meaning, etc


```python
alphas = np.arange(0, 1, .1)

def train_and_predict(alpha):

    nb_classifier = MultinomialNB(alpha=alpha)

    nb_classifier.fit(tfidf_train, y_train)

    pred = nb_classifier.predict(tfidf_test)

    score = metrics.accuracy_score(y_test, pred)
    return score


for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()
    
```

    Alpha:  0.0
    Score:  0.8813964610234337
    
    Alpha:  0.1
    Score:  0.8976566236250598
    
    Alpha:  0.2
    Score:  0.8938307030129125
    
    Alpha:  0.30000000000000004
    Score:  0.8900047824007652
    
    Alpha:  0.4


    /Users/Ocean/anaconda3/lib/python3.6/site-packages/sklearn/naive_bayes.py:472: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
      'setting alpha = %.1e' % _ALPHA_MIN)


    Score:  0.8857006217120995
    
    Alpha:  0.5
    Score:  0.8842659014825442
    
    Alpha:  0.6000000000000001
    Score:  0.874701099952176
    
    Alpha:  0.7000000000000001
    Score:  0.8703969392635102
    
    Alpha:  0.8
    Score:  0.8660927785748446
    
    Alpha:  0.9
    Score:  0.8589191774270684




```python
class_labels = nb_classifier.classes_

feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20], '\n')

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])

```

    FAKE [(-13.817639290604365, '0000'), (-13.817639290604365, '000035'), (-13.817639290604365, '0001'), (-13.817639290604365, '0001pt'), (-13.817639290604365, '000km'), (-13.817639290604365, '0011'), (-13.817639290604365, '006s'), (-13.817639290604365, '007'), (-13.817639290604365, '007s'), (-13.817639290604365, '008s'), (-13.817639290604365, '0099'), (-13.817639290604365, '00am'), (-13.817639290604365, '00p'), (-13.817639290604365, '00pm'), (-13.817639290604365, '014'), (-13.817639290604365, '015'), (-13.817639290604365, '018'), (-13.817639290604365, '01am'), (-13.817639290604365, '020'), (-13.817639290604365, '023')] 
    
    REAL [(-6.172241591175732, 'republicans'), (-6.126896127062493, 'percent'), (-6.115534950553315, 'political'), (-6.067024557833956, 'house'), (-5.9903983888515535, 'like'), (-5.986816295469049, 'just'), (-5.97418288622825, 'time'), (-5.964034477506528, 'states'), (-5.949002396420198, 'sanders'), (-5.844483857160232, 'party'), (-5.728156816243612, 'republican'), (-5.63452121120962, 'campaign'), (-5.5727798946931095, 'new'), (-5.515621480853161, 'state'), (-5.511414074572205, 'obama'), (-5.482207812723569, 'president'), (-5.455931002028523, 'people'), (-4.98170150128453, 'clinton'), (-4.5936919152219655, 'trump'), (-4.477148234163137, 'said')]





# ChatBot


```python
bot_template = "BOT : {0}"
user_template = "USER : {0}"

# Define a function that responds to a user's message: respond
def respond(message):
    # Concatenate the user's message to the end of a standard bot respone
    bot_message = "I can hear you! You said: " + message
    # Return the result
    return message

import time

# Define a function that sends a message to the bot: send_message
def send_message(message):
    # Print user_template including the user_message
    print(user_template.format(message))
    # Get the bot's response to the message
    response = respond(message)

    time.sleep(1.5) # artificial delay minicking natural
    print(bot_template.format(response))
```


```python
# Send a message to the bot
send_message("hello")
send_message("What did I say?")
```

    USER : hello
    BOT : hello
    USER : What did I say?
    BOT : What did I say?


## Personification
- chatbot not command line
- fun and use
- python module Smaltalk
- Simple: dict{key:response} , exact match-only
- Variable: response = dict{options}
- Asking Questions: input 


```python
import random

name = "Greg"
weather = "cloudy"

# Define a dictionary containing a list of responses for each message
responses = {
  "what's your name?": [
      "my name is {0}".format(name),
      "they call me {0}".format(name),
      "I go by {0}".format(name)
   ],
  "what's today's weather?": [
      "the weather is {0}".format(weather),
      "it's {0} today".format(weather)
    ],
  "default": ["default message"]
}

# Use random.choice() to choose a matching response
def respond(message):
    # Check if the message is in the responses
    if message in responses:
        # Return a random matching response
        bot_message = random.choice(responses[message])
    else:
        # Return a random "default" response
        bot_message = random.choice(responses["default"])
    return bot_message
```


```python
import random

responses_question = {'question': ["I don't know :(", 
                                   'you tell me!'],
                      'statement': 
                      ['tell me more!', 'why do you think that?',
                       'how long have you felt this way?',
                       'I find that extremely interesting',
                       'can you back that up?','oh wow!',
                       ':)']}
def respond(message):
    # Check for a question mark
    if message.endswith("?"):
        # Return a random question
        return random.choice(responses_question["question"])
    # Return a random statement
    return random.choice(responses_question["statement"])


# Send messages ending in a question mark
send_message("what's today's weather?")
send_message("what's today's weather?")

# Send messages which don't end with a question mark
send_message("I love building chatbots")
send_message("I love building chatbots")
```

    USER : what's today's weather?
    BOT : I don't know :(
    USER : what's today's weather?
    BOT : I don't know :(
    USER : I love building chatbots
    BOT : I find that extremely interesting
    USER : I love building chatbots
    BOT : tell me more!


### Using Regex to Match Pattern and Respond


```python
import re

rules = {'I want (.*)': 
             ['What would it mean if you got {0}',
              'Why do you want {0}',
              "What's stopping you from getting {0}"],
         'do you remember (.*)': 
             ['Did you think I would forget {0}',
              "Why haven't you been able to forget {0}",
              'What about {0}',
              'Yes .. and?'],
         'do you think (.*)': 
             ['if {0}? Absolutely.', 'No chance'],
         'if (.*)': 
             ["Do you really think it's likely that {0}",
              'Do you wish that {0}',
              'What do you think about {0}',
              'Really--if {0}']}

def match_rule(rules, message):
    response, phrase = "default", None
    
    for pattern, responses in rules.items():
        match = re.search(pattern, message)
        if match is not None:
            response = random.choice(responses)
            if '{0}' in response:
                phrase = match.group(1)
    return response, phrase
```


```python
print(match_rule(rules, "nice"))
```

    ('default', None)



```python
# Grammar, Pronouns change

def replace_pronouns(message):
    
    message = message.lower()
    if 'me' in message:
        return re.sub('me', 'you', message)
    if 'my' in message:
        return re.sub('my', 'your', message)
    if 'your' in message:
        return re.sub('your', 'my', message)
    if 'you' in message:
        return re.sub('you', 'me', message)
    
    return message

```


```python
print(replace_pronouns("my last birthday"))
print(replace_pronouns("when you went to Florida"))
print(replace_pronouns("I had my own castle"))
```

    your last birthday
    when me went to florida
    i had your own castle



```python
# Combining previous two functions

def respond(message):

    response, phrase = match_rule(rules, message)
    if '{0}' in response:

        phrase = replace_pronouns(phrase)
        # Include the phrase in the response
        response = response.format(phrase)
    return response


send_message("do you remember your last birthday")
send_message("do you think humans should be worried about AI")
send_message("I want a robot friend")
send_message("what if you could be anything you wanted")

```

    USER : do you remember your last birthday
    BOT : What about my last birthday
    USER : do you think humans should be worried about AI
    BOT : if humans should be worried about ai? Absolutely.
    USER : I want a robot friend
    BOT : What's stopping you from getting a robot friend
    USER : what if you could be anything you wanted
    BOT : Really--if me could be anything me wanted


# NLU - Sub-NLP
- Logic Loop
    1. "I'm look for a Mexico restaurant in the center of town"
    2. NLU model
    3. **intent** - restaurant_search
    4. **entities** - cuisine: Mexican, area: center
    5. Database
    6. "Sure! what about Pepe's Buritos on Main St?"


```python
keywords = {'goodbye': ['bye', 'farewell'],
 'greet': ['hello', 'hi', 'hey'],
 'thankyou': ['thank', 'thx']}

patterns = {}

for intent, keys in keywords.items():
    patterns[intent] = re.compile('|'.join(keys))
    
print(patterns)
```

    {'goodbye': re.compile('bye|farewell'), 'greet': re.compile('hello|hi|hey'), 'thankyou': re.compile('thank|thx')}



```python
# Define a function to find the intent of a message
def match_intent(message):
    matched_intent = None
    
    for intent, pattern in patterns.items():
        if pattern.search(message):
            matched_intent = intent
    return matched_intent

responses_intent = {'default': 'default message',
 'goodbye': 'goodbye for now',
 'greet': 'Hello you! :)',
 'thankyou': 'you are very welcome'}

def respond(message):

    intent = match_intent(message)
    key = "default"
    if intent in responses_intent:
        key = intent
    return responses_intent[key]


send_message("hello!")
send_message("bye byeee")
send_message("thanks very much!")
```

    USER : hello!
    BOT : Hello you! :)
    USER : bye byeee
    BOT : goodbye for now
    USER : thanks very much!
    BOT : you are very welcome



```python
def find_name(message):
    name = None
    # Create a pattern for checking if the keywords occur
    name_keyword = re.compile("name|call")
    # Create a pattern for finding capitalized words
    name_pattern = re.compile('[A-Z]{1}[a-z]*')
    if name_keyword.search(message):
        # Get the matching words in the string
        name_words = name_pattern.findall(message)
        if len(name_words) > 0:
            # Return the name if the keywords are present
            name = ' '.join(name_words)
    return name


def respond(message):

    name = find_name(message)
    if name is None:
        return "Hi there!"
    else:
        return "Hello, {0}!".format(name)


send_message("my name is David Copperfield")
send_message("call me Ishmael")
send_message("People call me Cassandra")

```

    USER : my name is David Copperfield
    BOT : Hello, David Copperfield!
    USER : call me Ishmael
    BOT : Hello, Ishmael!
    USER : People call me Cassandra
    BOT : Hello, People Cassandra!


# ML in Chatbot
- Predict(Intent)
- Many ways of Vectorisation of text
- Here word-vector to repr meaning of similar context
- **Open source of high quality word-vectors** 
## SpaCy module
- word-vector hundreds of elements
- Similarity measured by **angle** between vectors **distance**
    **Cosine Similarity**
    - 1 if same direction
    - 0 if orthogonal
    - -1 if opposite


```python
# dataset: flight booking system interaction from ATIS

sentences_demo = [' i want to fly from boston at 838 am and arrive in denver at 1110 in the morning',
 ' what flights are available from pittsburgh to baltimore on thursday morning',
 ' what is the arrival time in san francisco for the 755 am flight leaving washington',
 ' cheapest airfare from tacoma to orlando',
 ' round trip fares from pittsburgh to philadelphia under 1000 dollars',
 ' i need a flight tomorrow from columbus to minneapolis',
 ' what kind of aircraft is used on a flight from cleveland to dallas',
 ' show me the flights from pittsburgh to los angeles on thursday',
 ' all flights from boston to washington',
 ' what kind of ground transportation is available in denver',
 ' show me the flights from dallas to san francisco',
 ' show me the flights from san diego to newark by way of houston',
 ' what is the cheapest flight from boston to bwi',
 ' all flights to baltimore after 6 pm',
 ' show me the first class fares from boston to denver',
 ' show me the ground transportation in denver',
 ' all flights from denver to pittsburgh leaving after 6 pm and before 7 pm',
 ' i need information on flights for tuesday leaving baltimore for dallas dallas to boston and boston to baltimore',
 ' please give me the flights from boston to pittsburgh on thursday of next week',
 ' i would like to fly from denver to pittsburgh on united airlines',
 ' show me the flights from san diego to newark',
 ' please list all first class flights on united from denver to baltimore',
 ' what kinds of planes are used by american airlines',
 " i'd like to have some information on a ticket from denver to pittsburgh and atlanta",
 " i'd like to book a flight from atlanta to denver",
 ' which airline serves denver pittsburgh and atlanta',
 " show me all flights from boston to pittsburgh on wednesday of next week which leave boston after 2 o'clock pm",
 ' atlanta ground transportation',
 ' i also need service from dallas to boston arriving by noon',
 ' show me the cheapest round trip fare from baltimore to dallas']
```


```python
import spacy

nlp = spacy.load('en_core_web_lg') # 1G size of library

n_sentences_demo = len(sentences_demo)

embedding_dim = nlp.vocab.vectors_length

```


```python
X = np.zeros((n_sentences_demo, embedding_dim))

for idx, sentence in enumerate(sentences_demo):
    doc = nlp(sentence) # pass each sent to nlp obj
    X[idx, :] = doc.vector # pass .vector attr to row in X

```

## From Word-Vector to ML Intent and Classification
- 'msg' to intent recognition
- KNN is simple, Support Vector is better


```python
df_intent_train = pd.read_csv('Data_Folder/atis/atis_intents_train.csv', header=None)
df_intent_test = pd.read_csv('Data_Folder/atis/atis_intents_test.csv', header=None)

intent_X_train, intent_y_train = df_intent_train[:][1], df_intent_train[:][0]
intent_X_test, intent_y_test = df_intent_test[:][1], df_intent_test[:][0]                                                             
```


```python
X_train = np.zeros((len(intent_X_train), nlp.vocab.vectors_length))

for idx, sent in enumerate(intent_X_train):
    X_train[idx,:] = nlp(sent).vector

# long time......
```


```python
X_test = np.zeros
```


```python
# simple method using label as intent via cosine similarity in sent-vectors

from sklearn.metrics.pairwise import cosine_similarity

X_test = nlp(intent_X_test[0][1]).vector

scores = [cosine_similarity(X_train[i,:].reshape(-1,1), X_test.reshape(-1,1)) for i in range(len(intent_X_train))]

intent_y_train[np.argmax(scores)]

```




    'atis_flight'




```python
# using Full X_test vectors for SVM

X_test = np.zeros((len(intent_X_test), nlp.vocab.vectors_length))

for idx, sent in enumerate(intent_X_test):
    X_test[idx,:] = nlp(sent).vector
```


```python
# SVC

from sklearn.svm import SVC

clf = SVC(C=1)

clf.fit(X_train, intent_y_train)

y_pred = clf.predict(X_test)

n_correct = 0
for i in range(len(intent_y_test)):
    if y_pred[i] == intent_y_test[i]:
        n_correct += 1

print("Predicted {0} correctly out of {1} test examples".format(n_correct, len(intent_y_test)))

```




    SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



    Predicted 689 correctly out of 800 test examples


## Entity Extraction
- Unseen ER is tricky
- Generalisation - **pattern** and contextual cues:
    1. spelling
    2. capitalisation
    3. sequence of word pairs
- Pre-built NER
    1. places, dates, orgs, etc
- Roles
    - xxx from x to x ; xxx to x from x
    - `re.compile('.* from (.*) to (.*)')`
    - `re.compile('.* to (.*) from (.*)')`
### Dependency Parsing is complex topic
- spacy attr of token `token.ancestors` for parent token of word token



```python
# Define included entities
include_entities = ['DATE', 'ORG', 'PERSON']

# Define extract_entities()
def extract_entities(message):
    
    ents = dict.fromkeys(include_entities) # useful dict.fromkeys func

    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ in include_entities:
            # Save interesting entities
            ents[ent.label_] = ent.text
    return ents

print(extract_entities('friends called Mary who have worked at Google since 2010'))
print(extract_entities('people who graduated from MIT in 1999'))

```

    {'DATE': '2010', 'ORG': 'Google', 'PERSON': 'Mary'}
    {'DATE': '1999', 'ORG': 'MIT', 'PERSON': None}



```python
# SpaCy's powerful syntax parser to assign ROLES to entities in message

doc = nlp("let's see that jacket in red and some blue jeans")

colors = ['black', 'red', 'blue']
items = ['shoes', 'handback', 'jacket', 'jeans']

def entity_type(word):
    _type = None
    if word.text in colors:
        _type = "color"
    elif word.text in items:
        _type = "item"
    return _type

# Iterate over parents in parse tree until an item entity is found
def find_parent_item(word):
    # Iterate over the word's ancestors
    for parent in word.ancestors:
        # Check for an "item" entity
        if entity_type(parent) == "item":
            return parent.text
    return None

# For all color entities, find their parent item
def assign_colors(doc):
    # Iterate over the document
    for word in doc:
        # Check for "color" entities
        if entity_type(word) == "color":
            # Find the parent
            item =  find_parent_item(word)
            print("item: {0} has color : {1}".format(item, word))

# Assign the colors
assign_colors(doc)

```

    item: jacket has color : red
    item: jeans has color : blue


## Robust NLU with Rasa
- high-level API for intent recogition & entity extraction
- Based on spaCy, scikit-learn, other lib
- Built-in support for chatbot specific tasks
- data **json** file
### Special - predicting typo or unseen word
- `'intent_featurizer_ngrams'` : predictive of ngram in sub-vectors
- Ensure such config is required in context, i.e. enough data of such for learning

```python
# Import necessary modules
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

# Create args dictionary
args = {"pipeline": "spacy_sklearn"}

# Create a configuration and trainer
config = RasaNLUConfig(cmdline_args=args)
trainer = Trainer(config)

# Load the training data
training_data = load_data("./training_data.json")

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Try it out
print(interpreter.parse("I'm looking for a Mexican restaurant in the North of town"))
```

## Rasa
```python
# Import necessary modules
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

pipeline = [
    "nlp_spacy",
    "tokenizer_spacy",
    "ner_crf"
]

# Create a config that uses this pipeline
config = RasaNLUConfig(cmdline_args={"pipeline": pipeline})

# Create a trainer that uses this config
trainer = Trainer(config)

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Parse some messages
print(interpreter.parse("show me Chinese food in the centre of town"))
print(interpreter.parse("I want an Indian restaurant in the west"))
print(interpreter.parse("are there any good pizza places in the center?"))
```

## Virtual Assistance
- accessing SQL
- bad practice to SQL injection, using python syntax like {} .foramt() on query
- good practice: t= (area, price) ; c.execute("....", t)


```python
import sqlite3

conn = sqlite3.connect('Data_Folder/hotels.db')

conn
```




    <sqlite3.Connection at 0x12a847c70>




```python
c = conn.cursor()

c.execute("SELECT * FROM hotels WHERE area='south' and price='hi'")
```




    <sqlite3.Cursor at 0x12a8339d0>




```python
c.fetchall()
```




    [('Grand Hotel', 'hi', 'south', 5)]




```python
area, price = "south", "hi"
t = (area, price)

# key
c.execute('SELECT * FROM hotels WHERE area=? AND price=?', t)

print(c.fetchall())

```




    <sqlite3.Cursor at 0x12a8339d0>



    [('Grand Hotel', 'hi', 'south', 5)]


## Exploring DB with NL
- Logic
    1. using trained **Rasa interpreter** to **parser** message
    2. result = **entities dict**
    3. define params = {} storing key-val of entities
    4. feed query by filtering by params
    5. execute query with condition AND (or .join('query')
    6. Responses: default None; but result -> ...and or but...

```python
# Define find_hotels()
def find_hotels(params):
    # Create the base query
    query = 'SELECT * FROM hotels'
    # Add filter clauses for each of the parameters
    if len(params) > 0:
        filters = ["{}=?".format(k) for k in params]
        query += " WHERE " + " and ".join(filters)
    # Create the tuple of values
    t = tuple(params.values())
    
    # Open connection to DB
    conn = sqlite3.connect('hotels.db')
    # Create a cursor
    c = conn.cursor()
    # Execute the query
    c.execute(query, t)
    # Return the results
    return c.fetchall()
```

### Above find and match any range combination
```python
# Create the dictionary of column names and values
params = {"area": "south", "price":"lo"}

# Find the hotels that match the parameters
print(find_hotels(params))


# Define respond()
def respond(message):
    # Extract the entities
    entities = interpreter.parse(message)["entities"]
    # Initialize an empty params dictionary
    params = {}
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find hotels that match the dictionary
    results = find_hotels(params)
    # Get the names of the hotels and index of the response
    names = [r[0] for r in results]
    n = min(len(results),3)
    # Select the nth element of the responses array
    return responses[n].format(*names)

# Define respond()
def respond(message):
    # Extract the entities
    entities = interpreter.parse(message)["entities"]
    # Initialize an empty params dictionary
    params = {}
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find hotels that match the dictionary
    results = find_hotels(params)
    # Get the names of the hotels and index of the response
    names = [r[0] for r in results]
    n = min(len(results),3)
    # Select the nth element of the responses array
    return responses[n].format(*names)

print(respond("I want an expensive hotel in the south of town"))
```

## Incremental Slot Filling and Negation
- **memory-filled response** incrementally 
- Basic Memory - saving params in memory
- **negation** filtering response by negation or certainty
- tricky topic
- Negated entities - "no, not, etc" + NE 
    1. 'not sushi, maybe pizza?'


```python
# Define a respond function, taking the message and existing params as input
def respond(message, params):
    # Extract the entities
    entities = interpreter.parse(message)["entities"]
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find the hotels
    results = find_hotels(params)
    names = [r[0] for r in results]
    n = min(len(results), 3)
    # Return the appropriate response
    return responses[n].format(*names), params

# Initialize params dictionary
params = {}

# Pass the messages to the bot
for message in ["I want an expensive hotel", "in the north of town"]:
    print("USER: {}".format(message))
    response, params = respond(message, params)
    print("BOT: {}".format(response))
```

#### Basic Negation

```python
# Define negated_ents()
def negated_ents(phrase):
    # Extract the entities using keyword matching
    ents = [e for e in ["south", "north"] if e in phrase]
    # Find the index of the final character of each entity
    ends = sorted([phrase.index(e) + len(e) for e in ents])
    # Initialise a list to store sentence chunks
    chunks = []
    # Take slices of the sentence up to and including each entitiy
    start = 0
    for end in ends:
        chunks.append(phrase[start:end])
        start = end
    result = {}
    # Iterate over the chunks and look for entities
    for chunk in chunks:
        for ent in ents:
            if ent in chunk:
                # If the entity is preceeded by a negation, give it the key False
                if "not" in chunk or "n't" in chunk:
                    result[ent] = False
                else:
                    result[ent] = True
    return result  

# Check that the entities are correctly assigned as True or False
for test in tests:
    print(negated_ents(test[0]) == test[1])
```

```python
# Define the respond function
def respond(message, params, neg_params):
    # Extract the entities
    entities = interpreter.parse(message)["entities"]
    ent_vals = [e["value"] for e in entities]
    # Look for negated entities
    negated = negated_ents(message, ent_vals)
    for ent in entities:
        if ent["value"] in negated and negated[ent["value"]]:
            neg_params[ent["entity"]] = str(ent["value"])
        else:
            params[ent["entity"]] = str(ent["value"])
    # Find the hotels
    results = find_hotels(params, neg_params)
    names = [r[0] for r in results]
    n = min(len(results),3)
    # Return the correct response
    return responses[n].format(*names), params, neg_params

# Initialize params and neg_params
params = {}
neg_params = {}

# Pass the messages to the bot
for message in ["I want a cheap hotel", "but not in the north of town"]:
    print("USER: {}".format(message))
    response, params, neg_params = respond(message, params, neg_params)
    print("BOT: {}".format(response))
```

## Statefulness 
- Policy mapping user input with action
- Sophistication adding select action content-dependency
- additional memory, **state machine** e.g. traffic light (3 states)
    1. state dependency
    2. e-commerce - browsing, info, order completion, questions
    3. int used for states
- example: coffee ordering
    1. INIT state - order intent -> choose state
    2. ORDER -> take input from INIT and CHOOSE
    3. sequential dict()


```python
# Define the INIT state
INIT = 0

# Define the CHOOSE_COFFEE state
CHOOSE_COFFEE = 1

# Define the ORDERED state
ORDERED = 2

# Define the policy rules
policy = {
    (INIT, "order"): (CHOOSE_COFFEE, "ok, Columbian or Kenyan?"),
    (INIT, "none"): (INIT, "I'm sorry - I'm not sure how to help you"),
    (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!"),
    (CHOOSE_COFFEE, "none"): (CHOOSE_COFFEE, "I'm sorry - would you like Colombian or Kenyan?"),
}

# Create the list of messages
messages = [
    "I'd like to become a professional dancer",
    "well then I'd like to order some coffee",
    "my favourite animal is a zebra",
    "kenyan"
]

# Call send_message() for each message
state = INIT
for message in messages:    
    state = send_message(policy, state, message)
```

```python
# Define the states
INIT=0 
CHOOSE_COFFEE=1
ORDERED=2

# Define the policy rules dictionary
policy_rules = {
    (INIT, "ask_explanation"): (INIT, "I'm a bot to help you order coffee beans"),
    (INIT, "order"): (CHOOSE_COFFEE, "ok, Columbian or Kenyan?"),
    (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!"),
    (CHOOSE_COFFEE, "ask_explanation"): (CHOOSE_COFFEE, "We have two kinds of coffee beans - the Kenyan ones make a slightly sweeter coffee, and cost $6. The Brazilian beans make a nutty coffee and cost $5.")    
}

# Define send_messages()
def send_messages(messages):
    state = INIT
    for msg in messages:
        state = send_message(state, msg)

# Send the messages
send_messages([
    "what can you do for me?",
    "well then I'd like to order some coffee",
    "what do you mean by that?",
    "kenyan"
])
```

```python
# Define respond()
def respond(message, params, prev_suggestions, excluded):
    # Interpret the message
    parse_data = interpret(message)
    # Extract the intent
    intent = parse_data["intent"]["name"]
    # Extract the entities
    entities = parse_data["entities"]
    # Add the suggestion to the excluded list if intent is "deny"
    if intent == "deny":
        excluded.extend(prev_suggestions)
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])
    # Find matching hotels
    results = [
        r 
        for r in find_hotels(params, excluded) 
        if r[0] not in excluded
    ]
    # Extract the suggestions
    names = [r[0] for r in results]
    n = min(len(results), 3)
    suggestions = names[:2]
    return responses[n].format(*names), params, suggestions, excluded

# Initialize the empty dictionary and lists
params, suggestions, excluded = {}, [], []

# Send the messages
for message in ["I want a mid range hotel", "no that doesn't work for me"]:
    print("USER: {}".format(message))
    response, params, suggestions, excluded = respond(message, params, suggestions, excluded)
    print("BOT: {}".format(response))
```

## Question and Queuing Answers
- State machine building up memory and rules
- **complexity reduction** is needed
- e.g. coffee filter added in sales
    1. add additional states with policy on handling yes or no
    2. adding more states up complexity
    3. solution: **Pending actions** -> select action + pending_action (None)
    4. pending_action is saved in outer scope
    5. if 'yes' intent, pending, else none
    6. Pending state transitions -> authentication states -> request info -> transition to order state -> order

```python
# Define policy()
def policy(intent):
    # Return "do_pending" if the intent is "affirm"
    if intent == "affirm":
        return "do_pending", None
    # Return "Ok" if the intent is "deny"
    if intent == "deny":
        return "Ok", None
    if intent == "order":
        return "Unfortunately, the Kenyan coffee is currently out of stock, would you like to order the Brazilian beans?", "Alright, I've ordered that for you!"
```

#### Incorporate it into send_message() func

```python
# Define send_message()
def send_message(pending, message):
    print("USER : {}".format(message))
    action, pending_action = policy(interpret(message))
    if action == 'do_pending' and pending is not None:
        print("BOT : {}".format(pending))
    else:
        print("BOT : {}".format(action))
    return pending_action
    
# Define send_messages()
def send_messages(messages):
    pending = None
    for msg in messages:
        pending = send_message(pending, msg)

# Send the messages
send_messages([
    "I'd like to order some coffee",
    "ok yes please"
])
```

```python
# Define the states
INIT=0
AUTHED=1
CHOOSE_COFFEE=2
ORDERED=3

# Define the policy rules
policy_rules = {
    (INIT, "order"): (INIT, "you'll have to log in first, what's your phone number?", AUTHED),
    (INIT, "number"): (AUTHED, "perfect, welcome back!", None),
    (AUTHED, "order"): (CHOOSE_COFFEE, "would you like Columbian or Kenyan?", None),    
    (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!", None)
}

# Define send_messages()
def send_messages(messages):
    state = INIT
    pending = None
    for msg in messages:
        state, pending = send_message(state, pending, msg)

# Send the messages
send_messages([
    "I'd like to order some coffee",
    "555-12345",
    "kenyan"
])
```

```python
# Define chitchat_response()
def chitchat_response(message):
    # Call match_rule()
    response, phrase = match_rule(eliza_rules, message)
    # Return none is response is "default"
    if response == "default":
        return None
    if '{0}' in response:
        # Replace the pronouns of phrase
        phrase = replace_pronouns(phrase)
        # Calculate the response
        response = response.format(phrase)
    return response
```

```python
# Define send_message()
def send_message(state, pending, message):
    print("USER : {}".format(message))
    response = chitchat_response(message)
    if response is not None:
        print("BOT : {}".format(response))
        return state, None
    
    # Calculate the new_state, response, and pending_state
    new_state, response, pending_state = policy_rules[(state, interpret(message))]
    print("BOT : {}".format(response))
    if pending is not None:
        new_state, response, pending_state = policy_rules[pending]
        print("BOT : {}".format(response))        
    if pending_state is not None:
        pending = (pending_state, interpret(message))
    return new_state, pending

# Define send_messages()
def send_messages(messages):
    state = INIT
    pending = None
    for msg in messages:
        state, pending = send_message(state, pending, msg)

# Send the messages
send_messages([
    "I'd like to order some coffee",
    "555-12345",
    "do you remember when I ordered 1000 kilos by accident?",
    "kenyan"
])  

```

## Frontiers of dialogue tech
- many applications but not in chatbot
- context-relevant data needed
- Neural Conversational Model **Seq2seq**
    1. machine translation
    2. NN reads message, buiding up hidden vectors meaning
    3. reverse -> output sequence
    4. totally different

### No specified intent or etc, utterly data-driven
- not easy to integrate DB and API logic
- previous hand-crafted, seq2seq data-driven
- ML based:
    1. NLU
    2. Dialogue state manageuer
    3. API logic (connector to real world, DB)
    4. NL reponse generator
- 'Human pretend to be a bot: 'Wizard of Oz' technique
- RL - receives reward for successful conversation, improves over time

### Language generation
- practically, bot not recommended - better crafted than generated 
- but fun topic 
- NN trained can generate text from certain topic database, e.g. simpson scripts


```python
# Feed the 'seed' text into the neural network
seed = "i'm gonna punch lenny in the back of the"

# Iterate over the different temperature values
for temperature in [0.2, 0.5, 1.0, 1.2]:
    print("\nGenerating text with riskiness : {}\n".format(temperature))
    # Call the sample_text function
    print(sample_text(seed, temperature))
```

https://www.datacamp.com/community/tutorials/facebook-chatbot-python-deploy