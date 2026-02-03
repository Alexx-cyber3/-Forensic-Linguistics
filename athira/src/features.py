import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger_eng')

def extract_features(text):
    """
    Extracts stylometric features from a given text.
    Returns a dictionary of features.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Filter punctuation
    words_no_punct = [w for w in words if w.isalnum()]
    if not words_no_punct:
        return np.zeros(5) # Return zero vector if empty

    # 1. Lexical: Average Word Length
    avg_word_len = np.mean([len(w) for w in words_no_punct])
    
    # 2. Structural: Average Sentence Length (in words)
    avg_sent_len = np.mean([len(word_tokenize(s)) for s in sentences])
    
    # 3. Syntactic: POS Tag Ratios
    # POS tags: NN (Noun), VB (Verb), JJ (Adjective)
    pos_tags = nltk.pos_tag(words_no_punct)
    counts = {'NN': 0, 'VB': 0, 'JJ': 0}
    
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            counts['NN'] += 1
        elif tag.startswith('VB'):
            counts['VB'] += 1
        elif tag.startswith('JJ'):
            counts['JJ'] += 1
            
    total_words = len(words_no_punct)
    noun_ratio = counts['NN'] / total_words
    verb_ratio = counts['VB'] / total_words
    adj_ratio = counts['JJ'] / total_words
    
    # Feature Vector: [AvgWordLen, AvgSentLen, NounRatio, VerbRatio, AdjRatio]
    features = [avg_word_len, avg_sent_len, noun_ratio, verb_ratio, adj_ratio]
    return np.array(features)

def get_feature_names():
    return ["AvgWordLen", "AvgSentLen", "NounRatio", "VerbRatio", "AdjRatio"]
