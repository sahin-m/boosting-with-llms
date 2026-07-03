#Copyright 2026 Mücahit Sahin
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


#!pip install spacy==2.3.7
#!python -m spacy download en_core_web_lg

import re
import numpy as np
import pandas as pd
import os
import spacy
from nltk.corpus import words
import torch
import string
import unicodedata
import transformers
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from scipy.spatial.distance import cosine
import fasttext
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

# disable BERT warnings
from transformers import logging
logging.set_verbosity_error()


# Load the English NLP model from spaCy
nlp = spacy.load("en_core_web_lg")



def normalize_text(text):
    # Step 1: Replace any punctuation from string.punctuation with a space
    text = re.sub(f"[{string.punctuation}]", " ", text)

    # Step 2: Remove extra whitespaces 
    text = ' '.join(text.split())
    
    # Step 3: Convert text to lowercase
    text = text.lower()

    # Step 4: Remove any remaining non-alphabetic characters except spaces
    normalized_text = re.sub(r'[^A-Za-z0-9 ]+', '', text)

    
    # Step 5: Stopword removal
    #doc = nlp(text)
    #text_no_stopwords = " ".join([token.text for token in doc if not token.is_stop])  # Remove stopwords
    
    # Step 6: Remove punctuation 
    #doc_no_punct = nlp(text_no_stopwords)
    #tokens_no_punct = " ".join([token.text for token in doc_no_punct if not token.is_punct])

    # ----------------
    # doc = nlp(text)
    # tokens_no_punct = " ".join([token.text for token in doc if not token.is_punct])
    # # ID is treatet was two tokens but we want to keep it as one token
    # if 'i d' in tokens_no_punct:
    #     tokens_no_punct = tokens_no_punct.replace('i d', 'id')
    # ----------------

    # Step 7: Lemmatization
    #doc = nlp(tokens_no_punct)
    #tokens_lemmas = [token.lemma_ for token in doc]
    #normalized_text = " ".join(tokens_lemmas)

    # Step 8: Remove any remaining non-alphabetic characters except spaces
    #normalized_text = re.sub(r'[^A-Za-z ]+', '', normalized_text)

    
    return normalized_text



class BigramVectorizer:
    def __init__(self):
        self.bigrams = {}
        self.normalize = False
    
    def fit(self, df, normalize, top_k=None):
        self.normalize = normalize
        # Generate the bigrams and map them to unique values
        self.bigrams = self._generate_bigrams(df, top_k)
        return self

    def transform(self, df):
        transformed_df = df.copy()
        feature_names = df['Attribute_name'].astype(str).tolist()
    
        # Initialize dictionary to hold counts for each bigram column
        bigram_columns = {col: [] for col in self.bigrams.values()}
    
        # Loop through each feature name
        for feature in feature_names:
            if self.normalize:
                feature = normalize_text(feature)
    
            for bigram, col in self.bigrams.items():
                # Count non-overlapping occurrences of each bigram
                count = feature.count(bigram)
                bigram_columns[col].append(count)
    
        # Create DataFrame from the collected counts
        bigram_df = pd.DataFrame(bigram_columns)
        bigram_df.columns = bigram_df.columns.astype(str)
        
        # Concatenate with original
        transformed_df = pd.concat([transformed_df, bigram_df], axis=1)
    
        return transformed_df
    
    
    def _generate_bigrams(self, df, top_k):
        bigram_counter = Counter()
        feature_names = df['Attribute_name'].astype(str).values.tolist()

        if not top_k:
            top_k = len(feature_names)
    
        # Iterate over each feature name in the list
        for feature in feature_names:
            if self.normalize:
                feature = normalize_text(feature)
            else:
                feature = feature.lower()
                
            # Generate the 2-character bigrams from the normalized feature name
            for i in range(len(feature) - 1):
                bigram = feature[i:i+2]  # Extract the 2-character bigram
    
                if self.normalize:
                    # Only add bigrams that have two useful characters
                    if self._is_useful_bigram(bigram):
                        bigram_counter[bigram] += 1  # Count the occurrence of this bigram
                else:
                    bigram_counter[bigram] += 1  # Count the occurrence of this bigram
    
        # Get the top k most common bigrams (return only the bigrams, not the counts)
        bigrams = [bigram for bigram, _ in bigram_counter.most_common(top_k)]
        # Create dict with mapping from bigram to column number
        bigrams = {bigram: idx + 1 for idx, bigram in enumerate(bigrams, start=-1)}
        
        return bigrams

    # Function to check if a character is an accented letter (e.g., 'ä', 'õ')
    def _is_accented_letter(self, c):
        return not c.isascii()
        
    # Function to check if a 2-character string has only one useful piece of information
    def _is_useful_bigram(self, s):
        
        # Define useful characters (letters and special characters)
        letters = string.ascii_letters
        punctuations = string.punctuation
        digits = string.digits
        
        # Exclude if the first character is a space and the second is a digit or punctuation
        if (s[0] == ' ' and (s[1] in digits or s[1] in punctuations)) or \
           (s[1] == ' ' and (s[0] in digits or s[0] in punctuations)):
            return False
    
        # Exclude bigrams with a space followed by punctuation
        if (s[0] == ' ' and s[1] in punctuations) or \
           (s[1] == ' ' and s[0] in punctuations):
            return False
        
        # Exclude if one character is a space and the other is a useful character
        if (s[1] == ' ' and s[0] in letters) or \
           (s[0] == ' ' and s[1] in letters):
            return False
        
        # Exclude if the first character is a space and the second is a non-alphanumeric character
        if (s[0] == ' ' and not s[1].isalnum()) or \
           (s[1] == ' ' and not s[0].isalnum()):
            return False
    
        # Exclude if one character is a space and the other is an accented letter
        if (s[0] == ' ' and self.is_accented_letter(s[1])) or \
           (s[1] == ' ' and self.is_accented_letter(s[0])):
            return False
        
        # Exclude bigrams where a punctuation is followed by a lowercase letter (e.g., ';i', '"p')
        if (s[0] in punctuations and s[1] in letters) or \
           (s[1] in punctuations and s[0] in letters):
            return False
        
        # Exclude if one character is a space and the other is an accented letter
        if (s[0] == ' ' and _is_accented_letter(s[1])) or \
           (s[1] == ' ' and _is_accented_letter(s[0])):
            return False
    
        # Exclude if one character is a number and the other is a letter
        if (s[0] in digits and s[1] in letters) or \
           (s[1] in digits and s[0] in letters):
            return False
    
        # Exclude cases where both characters are digits
        if (s[0] in digits and s[1] in digits) or \
           (s[1] in digits and s[0] in digits):
            return False 
        
        # Exclude cases where both characters are punctuation (e.g., '";')
        if (s[0] in punctuations and s[1] in punctuations) or \
           (s[1] in punctuations and s[0] in punctuations):
            return False 
        
        return True



class WordVectorizer:
    def __init__(self):
        self.words = {}
        self.normalize = False
        self.english_vocab = set(words.words())

    def fit(self, df, normalize, top_k=None):
        self.normalize = normalize
        self.words = self._generate_words(df, top_k)
        return self

    def transform(self, df):
        transformed_df = df.copy()
        feature_names = df['Attribute_name'].astype(str).tolist()
        word_columns = {col: [] for col in self.words.values()}

        # Loop through each feature name
        for feature in feature_names:
            if self.normalize:
                feature = normalize_text(feature)
                
            words = feature.split()
            word_counts = Counter(words)

            for word, col in self.words.items():
                word_columns[col].append(word_counts.get(word, 0))   

        word_df = pd.DataFrame(word_columns)
        word_df.columns = word_df.columns.astype(str)

        transformed_df = pd.concat([transformed_df, word_df], axis=1)
        return transformed_df
        

    def _generate_words(self, df, top_k):
        word_counter = Counter()
        feature_names = df['Attribute_name'].astype(str).tolist()

        if not top_k:
            top_k = len(feature_names)

        for feature in feature_names:
            if self.normalize:
                feature = normalize_text(feature)
            else:
                feature = feature.lower()

            words = feature.split()
            for word in words:
                if self._is_useful_word(word):
                    word_counter[word] += 1

        most_common_words = [word for word, _ in word_counter.most_common(top_k)]
        return {word: idx + 1 for idx, word in enumerate(most_common_words, start=-1)}
        

    def _is_useful_word(self, word):
        letters = string.ascii_letters
        punctuations = string.punctuation
        digits = string.digits

        # Exclude short words (length 1 or less)
        if len(word) <= 1:
            return False

        # Exclude words that are only digits or punctuation
        if all(char in digits for char in word):
            return False
        if all(char in punctuations for char in word):
            return False

        # Exclude words that are mixed digits and letters
        if any(c in digits for c in word) and any(c in letters for c in word):
            return False

        # Exclude if it contains mostly non-ascii characters (accented or symbols)
        if sum(1 for c in word if not c.isascii()) > len(word) // 2:
            return False

        # Exclude words that are mostly punctuation
        if sum(1 for c in word if c in punctuations) > len(word) // 2:
            return False

        # Check if the word has any meaning
        if not word.lower() in self.english_vocab:
            return False

        return True



class TfidfNgramVectorizer:
    def __init__(self):
        self.normalize = False
        self.vectorizer = None
        self.features = {}

    def fit(self, df, normalize=False, top_k=None, ngram_range=(2, 2)):
        self.normalize = normalize
        feature_names = df['Attribute_name'].astype(str).tolist()
        
        if self.normalize:
            feature_names = [normalize_text(f) for f in feature_names]

        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=top_k if top_k else 500
        )
        self.vectorizer.fit(feature_names)
        features = self.vectorizer.get_feature_names_out()
        for i in range(len(features)):
            self.features[i] = features[i] 

        return self

    def transform(self, df):
        transformed_df = df.copy()
        feature_names = df['Attribute_name'].astype(str).tolist()

        if self.normalize:
            feature_names = [normalize_text(f) for f in feature_names]

        tfidf_matrix = self.vectorizer.transform(feature_names)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[str(t) for t in self.features.keys()])
        tfidf_df.columns = tfidf_df.columns.astype(str)

        transformed_df = pd.concat([transformed_df, tfidf_df], axis=1)
        return transformed_df
    


class BERTVectorizer:
    def __init__(self):
        self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    def transform(self, df):
        feature_names = df['Attribute_name'].fillna('nan').values.tolist()

        # Normalize the names
        feature_names = [normalize_text(f) for f in feature_names]

        # Get the embeddings
        embeddings = [self._get_embedding(feature_name) for feature_name in feature_names]
        
        # Embeddings are tensors, covnert them numpy
        embeddings = np.stack([tensor.numpy() for tensor in embeddings])
        
        # Convert the list of embeddings into a DataFrame
        embedding_df = pd.DataFrame(embeddings)
        
        # Rename the columns
        embedding_df.columns = [f'BERT_embedding_dim_{dim}' for dim in range(768)]

        # Concatenate the features with the original DataFrame
        df_new = pd.concat([df, embedding_df], axis=1)

        return df_new
        
        
    # Function to get word embeddings
    def _get_embedding(self, text):
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=True)
        
        # Get the model output (embeddings)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract embeddings and attention mask (exclude padding tokens)
        word_embeddings = outputs.last_hidden_state[0]
        attention_mask = inputs['attention_mask'][0]

        # Mask the embeddings: keep only embeddings where mask == 1
        valid_embeddings = word_embeddings[attention_mask.bool()]   # shape: [num_valid_tokens, hidden_size]
    
        # Initialize with ones
        embeddings = torch.zeros(valid_embeddings.size(1))            # shape: [hidden_size]
    
        # Element-wise addition across all valid token embeddings
        for vec in valid_embeddings:
            embeddings += vec
    
        return embeddings
        

    # Function to compute cosine similarity between two embeddings (testing purposes)
    def compute_cosine_similarity(self, embedding1, embedding2):
        
        # Calculate cosine similarity
        similarity = 1 - cosine(embedding1, embedding2)
        return round(similarity, 4)



class GloVeVectorizer:
    def transform(self, df):
        feature_names = df['Attribute_name'].fillna('nan').values.tolist()

        # Normalize the names
        feature_names = [normalize_text(f) for f in feature_names]

        # Get the embeddings
        embeddings = [self._get_embedding(feature_name) for feature_name in feature_names]
        embeddings = np.array([embedding[0] for embedding in embeddings])

        # Convert the list of embeddings into a DataFrame
        embedding_df = pd.DataFrame(embeddings)
        
        # Rename the columns
        embedding_df.columns = [f'GloVe_embedding_dim_{dim}' for dim in range(300)]

        # Concatenate the features with the original DataFrame
        df_new = pd.concat([df, embedding_df], axis=1)

        return df_new
        
        
    # Function to get word embeddings
    def _get_embedding(self, text):
        # Split the text into individual words to sum up embeddings for single words
        words = [token.text for token in nlp(text)]

        # Initialize the embeddings sum
        embeddings = np.zeros((1, 300))

        # Sum the embeddings for each word
        for word in words:
            embedding = nlp(word).vector.reshape(1, -1)
            embeddings += embedding  # add embeddings

        return embeddings
        

    # Function to compute cosine similarity between two embeddings (testing purposes)
    def compute_cosine_similarity(self, embedding1, embedding2):
        
        # Calculate cosine similarity
        similarity = 1 - cosine(embedding1, embedding2)
        return round(similarity, 4)



class SentenceTFVectorizer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load a pretrained SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def transform(self, df):
        # Fill missing values and convert to a list of strings
        feature_names = df['Attribute_name'].fillna('nan').astype(str).tolist()

        # Normalize the names
        feature_names = [normalize_text(f) for f in feature_names]

        # Get sentence embeddings
        embeddings = self.model.encode(feature_names, convert_to_numpy=True)

        # Convert the embeddings into a DataFrame
        embedding_df = pd.DataFrame(embeddings)

        # Rename the columns
        embedding_df.columns = [f'SentenceTF_embedding_dim_{dim}' for dim in range(embedding_df.shape[1])]

        # Concatenate the features with the original DataFrame
        df_new = pd.concat([df, embedding_df], axis=1)

        return df_new
        

    # Function to compute cosine similarity between two embeddings (testing purposes)
    def compute_cosine_similarity(self, embedding1, embedding2):
        
        # Calculate cosine similarity
        similarity = 1 - cosine(embedding1, embedding2)
        return round(similarity, 4)




class FastTextVectorizer:
    def __init__(self, model_repo="facebook/fasttext-en-vectors", filename="model.bin"):
        try:
            # Check if already in cache
            model_path = hf_hub_download(repo_id=model_repo, filename=filename, local_files_only=True)
        except (EntryNotFoundError, LocalEntryNotFoundError):
            # If not cached, download it
            model_path = hf_hub_download(repo_id=model_repo, filename=filename)

        self.model = fasttext.load_model(model_path)
        self.vector_size = self.model.get_dimension()
            

    def transform(self, df):
        # Extract and normalize feature names
        feature_names = df['Attribute_name'].fillna('nan').astype(str).tolist()
        feature_names = [normalize_text(f) for f in feature_names]

        # Compute averaged FastText embeddings
        embeddings = [self._get_embedding(name) for name in feature_names]
        embeddings = np.array([embedding[0] for embedding in embeddings])

        # Convert to DataFrame
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.columns = [f'FastText_embedding_dim_{i}' for i in range(self.vector_size)]

        # Concatenate with original DataFrame
        df_new = pd.concat([df, embedding_df], axis=1)
        return df_new


    def _get_embedding(self, text):
        # Split the text into individual words to sum up embeddings for single words
        words = [token.text for token in nlp(text)]

        embeddings = np.zeros((1, self.vector_size))

        for word in words:
            embedding = self.model.get_word_vector(word)
            embeddings += embedding  # add embeddings

        return embeddings
            

    # Function to compute cosine similarity between two embeddings (testing purposes)
    def compute_cosine_similarity(self, embedding1, embedding2):
        
        # Calculate cosine similarity
        similarity = 1 - cosine(embedding1, embedding2)
        return round(similarity, 4)