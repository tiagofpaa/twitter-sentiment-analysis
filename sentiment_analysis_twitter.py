#!/usr/bin/env python
# coding: utf-8

# # Imports
# 
# Zona em que são definidos os imports necessários.

# In[1]:


import json
import re
import csv
import itertools
import nltk
import collections
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import util
import nltk.classify
from nltk.classify import maxent, naivebayes, svm
from sklearn.svm import LinearSVC


# # Função com as métricas

# In[2]:


def metrics(true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred):
    
    print("True positives: {}".format(true_positive))
    print("True Negatives: {}".format(true_negative))
    print("False Positives: {}".format(false_positive))
    print("False Negatives: {}".format(false_negative))

    print("\nTweets Positivos: {}".format(pos_real))
    print("Tweets Positivos preditados: {}".format(pos_pred))
    print("Tweets Negativos: {}".format(neg_real))
    print("Tweets Negativos preditados: {}".format(neg_pred))
    
    print("\nPredicoes:")
    print("\tPredicoes correctas: {}".format(true_positive + true_negative))
    print("\tPredicoes erradas: {}".format(false_negative + false_positive))
    
    accuracy = (true_positive + true_negative)/(true_positive + true_negative+false_negative + false_positive)*100
    precision = true_positive/(true_positive+false_positive)*100
    recall = true_positive/(true_positive+false_negative)*100
    f1 = (2 * precision * recall)/(precision  + recall)

    print("\nMetricas:")
    print("\tAccuracy: {}".format(accuracy))
    print("\tPrecision: {}".format(precision))
    print("\tRecall: {}".format(recall))
    print("\tF-Measure: {}".format(f1))
    


# # Preparação dos dados e criação de uma baseline
# 
# O conjunto de dados escolhido foi o "Tweets_EN_sentiment.json".
# Nesta fase iremos realizar a preparação dos dados e a criação de uma baseline, para que se possa comparar resultados posteriores.
# Foi detectado um desbalanceamento dos dados, na qual existem muito mais tweets positivos do que negativos, o que provocaria resultados desajustados à realidade, por exemplo: classificar todos os tweets de teste como sendo positivos e e por isso obter resultados com percentagens bastante elevadas, pelo facto de existirem mais tweets positivos do que negativos.
# Portanto foram usados todos os tweets negativos, que estão em menor número, e foram o número de tweets positivos igual ao número de tweets negativos.
# De forma a ter um conjunto de dados completamente aleatório em tipo de sentimentos, foram baralhados os dados de igual forma.
# Ir-se-á utilizar as primeiras linhas, 80% mais propriamente, para treino e as últimas para teste, neste caso as restantes 20%. A ferramenta utilizada para fazer Análise de Sentimento diretamente aos tweets foi o TextBlob.

# In[3]:


tweets_unbalanced = []
for tweet in open("../TM/data/en/Tweets_EN_sentiment.json", "r"):
    tweets_unbalanced.append(json.loads(tweet))

count_neg = 0
for tweet in tweets_unbalanced:
    if tweet["class"] == "neg":
        count_neg += 1

tweets = []
count_pos = 0
count_neg_equal = 0
for tweet in tweets_unbalanced:
    number_tweet = tweet['tweet']
    text = tweet["text"]
    sentiment = tweet["class"]
    if tweet["class"] == "neg" and count_neg != count_neg_equal:
        count_neg_equal += 1
        tweet = {'tweet': number_tweet,'text': text, 'class': sentiment}
        tweets.append(tweet)
    if tweet["class"] == "pos" and count_pos < count_neg:
        count_pos += 1
        tweet = {'tweet': number_tweet, 'text': text, 'class': sentiment}
        tweets.append(tweet)
        
random.shuffle(tweets)


# Foi criada uma lista de treino com 80% dos dados e uma lista de teste com os 20% restantes.
# Para além disso verificou-se quantos tweets positivos e negativos foram utilizados para treino e teste.

# In[4]:


def train_test_lists(string, tweets, with_prints):
    if with_prints:
        print("--- " + string + " ---")
        print("\nTweets totais : {}".format(len(tweets)))
    
    train_perc = 0.8
    test_perc = 1 - train_perc
    train_size = round(len(tweets)*train_perc)
    test_size = round(len(tweets)*test_perc)
    train_list = tweets[:train_size]
    test_list = tweets[train_size:]

    positive_train = 0
    negative_train = 0

    for tweet in train_list:
        if tweet["class"] == "pos":
            positive_train += 1
        else:
            negative_train += 1

    if with_prints:
        print("\nTreino:")    
        print("\tPositivos: {}".format(positive_train))
        print("\tNegativos: {}".format(negative_train))
        print("\tTotal: {}".format(train_size))
        
        
    positive_test = 0
    negative_test = 0

    for tweet in test_list:
        if tweet["class"] == "pos":
            positive_test += 1
        else:
            negative_test += 1
    
    if with_prints:
        print("\nTeste:")    
        print("\tPositivos: {}".format(positive_test))
        print("\tNegativos: {}".format(negative_test))
        print("\tTotal: {}".format(test_size))
    
    return train_list, test_list
    
train_list, test_list = train_test_lists("Fase inicial", tweets, True)


# A ferramente utilizada para análise de sentimento diretamente a um texto foi o TextBlob.
# Para esta classificação inicial foi utilizada a Polarity do TextBlob, tal como o sentimento de cada tweet já fornecido pelo ficheiro "Tweets_EN_sentiment.json".
# Tendo em conta que existe um maior número de tweets considerados positivos, para criar uma maior equidade, os tweets classificados como neutro (polarity = 0 do TextBlob) passam a ser classificados como tweets negativos.

# In[5]:


true_positive = 0
true_negative = 0 
false_positive = 0
false_negative = 0
real_values = []
pred_values = []
pos_real = 0
pos_pred = 0
neg_real = 0
neg_pred = 0


for tweet in test_list:
    polarity = TextBlob(tweet["text"]).sentiment.polarity
    sentiment = tweet["class"]
      
    if sentiment == "pos" and polarity > 0:
        true_positive += 1
    if sentiment == "neg" and polarity < 0:
        true_negative += 1
    if not(sentiment == "pos" and polarity > 0) and not(sentiment == "neg" and polarity < 0):  
        if polarity > 0:
            false_positive += 1
        else:
            false_negative += 1
            
    if sentiment == 'pos':
        real_values.append(1)
        pos_real += 1
    if sentiment == 'neg':
        real_values.append(0)
        neg_real += 1
    if polarity > 0: 
        pred_values.append(1)
        pos_pred += 1
    if polarity <= 0:
        pred_values.append(0)
        neg_pred += 1

        
print("--- BASELINE ---\n\n")
metrics(true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred)


# # Retweets
# 
# Função que remove todos os tweets repetidos, i.e. Retweets, com a abreviação "RT".

# In[6]:


def retweets_treatemant(tweets):
    for tweet in tweets:
        number_tweet = tweet['tweet']
        words = TweetTokenizer().tokenize(tweet['text'])
        for word in words:
            if word == "RT":
                for i in range(len(tweets)):
                    if tweets[i]['tweet'] == number_tweet:
                        del tweets[i]
                        break
                        
    train_list, test_list = train_test_lists("Fase após remoção de Retweets", tweets, True)
    
    return train_list, test_list

train_list, test_list = retweets_treatemant(tweets)


# # Tratamento dos dados
# 
# Função para o tratamento dos dados.

# In[7]:


def data_treatment(tweet, with_punctuation, with_stop_words, with_pos_tag):
    tweet = abbreviations_treatment(tweet)
    tweet = emoticons_treatment(tweet)
    tweet = hashtag_treatment(tweet)
    tweet = url_treatment(tweet)
    tweet = usernames_treatment(tweet)
    tweet = numbers_treatment(tweet)
    tweet = money_treatment(tweet)
    tweet = time_treatment(tweet)
    tweet = lower_case_treatment(tweet)
    tweet = repetead_characters(tweet)
    if with_pos_tag:
        with_punctuation = True
        with_stop_words = True
    if not with_stop_words and not with_punctuation:
        tweet = stop_words_treatment(tweet)
        tweet = punctuation_symbols_treatment(tweet)
        return tweet
    if with_stop_words and not with_punctuation:
        tweet = punctuation_symbols_treatment(tweet)
        return tweet
    if not with_stop_words and with_punctuation:
        tweet = stop_words_treatment(tweet)
        return tweet
    if with_stop_words and with_punctuation:
        return tweet


# # Hashtags
# 
# Função que remove o símbolo de hashtag, substituindo pela palavra "hashtag", mantendo a palavra.
# Exemplo: "#Portugal" passa a "hashtag Portugal".

# In[8]:


def remove_hashtag(word):
    return word.replace("#", "hashtag ")


def hashtag_treatment(tweet):
    correct_tweet = tweet
    list_matches = [i.group(0) for i in re.finditer(r"\S*[#]\S*", tweet)]
    for word in list_matches:
        correct_tweet = correct_tweet.replace(word,remove_hashtag(word))
    return correct_tweet


# # URL's
# 
# Função que remove todos os URL.

# In[9]:


def url_treatment(tweet):
    tweet = re.sub(r'''(?:(?:https?|ftp):|(?:https?|ftp):\/\/|\b(?:[a-z\d]+\.))(?:(?:[^\s()<>]+|\((?:[^\s()<>]+|(?:\([^\s()<>]+\)))?\))+(?:\((?:[^\s()<>]+|(?:\(?:[^\s()<>]+\)))?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))?''', "", tweet)
    tweet = re.sub(" +"," ",tweet)
    return tweet


# # Pontuação
# 
# Função que remove toda a pontuação e símbolos.

# In[10]:


def punctuation_symbols_treatment(tweet):
    tweet = re.sub(r"[^\w\s]", " ", tweet)
    tweet = re.sub(" +"," ",tweet)
    tweet = re.sub(" +$","",tweet)
    tweet = re.sub("^ +","",tweet)
    return tweet


# # Nomes de utilizadores
# 
# Função que remove todos os nomes de utilizadores que comecem com "@".

# In[11]:


def usernames_treatment(tweet):
    tweet = re.sub(r"@+\w+", "", tweet)
    tweet = re.sub(" +"," ",tweet)
    return tweet


# # Minúsculas
# 
# Função que altera todos os caracteres do tweet para minúsculas.

# In[12]:


def lower_case_treatment(tweet):
    return tweet.lower()


# # Números
# 
# Função que remove todos os números, substituindo pela palavra "number".
# Exemplo: "10" passa a "number"

# In[13]:


def numbers_treatment(tweet):
    tweet = re.sub(r"(?<=\s)\d+(?=\s|$)|(?<=\s)\d+[.,]+\d+(?=\s|$)", "number", tweet)
    list_with_punctuation = [i.group(0) for i in re.finditer(r"(?<=\s)\d+[.,;:?!\-*\/+=%](?=\s|$)|(?<=\s)\d+[.,]+\d+[.,;:?!\-*\/+=%](?=\s|$)", tweet)]
    correct_tweet = tweet
    for word_punctuation in list_with_punctuation:
        punctuation = re.findall(r'[.,;:?!\-*\/+=%]', word_punctuation)
        correct_tweet = correct_tweet.replace(word_punctuation,"number" + punctuation[0])
    if re.compile("^\d+[.,]+\d\s|^\d\s").match(correct_tweet):
        word = re.findall(r'^\d+[.,]+\d\s|^\d\s', correct_tweet)
        word = word[0].strip()
        correct_tweet = correct_tweet.replace(word,"number")
    if re.compile("^\d+[.,]+\d+[.,;:?!\-*\/+=%]|^\d+[.,;:?!\-*\/+=%]").match(correct_tweet):
        start_with_punctuation = re.findall(r'^\d+[.,]+\d+[.,;:?!\-*\/+=%]|^\d+[.,;:?!\-*\/+=%]', correct_tweet)
        word = re.findall(r'\d+[.,]+\d+|\d+', start_with_punctuation[0])
        correct_tweet = correct_tweet.replace(word[0],"number")
    return correct_tweet
    


# # Dinheiro
# 
# Função que remove referencias a dinheiro, substituindo pela palavra "money". Exemplo: "$10" passa a "money"

# In[14]:


def money_treatment(tweet):
    tweet = re.sub(r"(?<=\s)[$]+\d+(?=\s|$)|(?<=\s)[$]+\d+[.,]+\d+(?=\s|$)|(?<=\s)\d+[$]+(?=\s|$)|(?<=\s)\d+[.,]+\d+[$]+(?=\s|$)|(?<=\s)[€]+\d+(?=\s|$)|(?<=\s)[€]+\d+[.,]+\d+(?=\s|$)|(?<=\s)\d+[€]+(?=\s|$)|(?<=\s)\d+[.,]+\d+[€]+(?=\s|$)", "money", tweet)
    list_with_punctuation = [i.group(0) for i in re.finditer(r"(?<=\s)[$]+\d+[.,;:?!\-*\/+](?=\s|$)|(?<=\s)[$]+\d+[.,]+\d+[.,;:?!\-*\/+](?=\s|$)|(?<=\s)\d+[$]+[.,;:?!\-*\/+](?=\s|$)|(?<=\s)\d+[.,]+\d+[$]+[.,;:?!\-*\/+](?=\s|$)|(?<=\s)[€]+\d+[.,;:?!\-*\/+](?=\s|$)|(?<=\s)[€]+\d+[.,]+\d+[.,;:?!\-*\/+](?=\s|$)|(?<=\s)\d+[€]+[.,;:?!\-*\/+](?=\s|$)|(?<=\s)\d+[.,]+\d+[€]+[.,;:?!\-*\/+](?=\s|$)", tweet)]
    correct_tweet = tweet
    for word_punctuation in list_with_punctuation:
        punctuation = re.findall(r'[.,;:?!\-*\/+]', word_punctuation)
        correct_tweet = correct_tweet.replace(word_punctuation,"money" + punctuation[0])
    if re.compile("^[$]+\d+[.,]+\d\s|^\d+[.,]+\d+[$]\s|^[€]+\d+[.,]+\d\s|^\d+[.,]+\d+[€]\s|^[$]+\d+\s|^\d+[$]\s|^[€]+\d\s|^\d+[€]\s").match(correct_tweet):
        word = re.findall(r'^[$]+\d+[.,]+\d\s|^\d+[.,]+\d+[$]\s|^[€]+\d+[.,]+\d\s|^\d+[.,]+\d+[€]\s|^[$]+\d+\s|^\d+[$]\s|^[€]+\d\s|^\d+[€]\s', correct_tweet)
        word = word[0].strip()
        correct_tweet = correct_tweet.replace(word,"money")
    if re.compile("^[$]+\d+[.,;:?!\-*\/+]|^[$]+\d+[.,]+\d+[.,;:?!\-*\/+]|^\d+[$]+[.,;:?!\-*\/+]|^\d+[.,]+\d+[$]+[.,;:?!\-*\/+]|^[€]+\d+[.,;:?!\-*\/+]|^[€]+\d+[.,]+\d+[.,;:?!\-*\/+]|^\d+[€]+[.,;:?!\-*\/+]|^\d+[.,]+\d+[€]+[.,;:?!\-*\/+]").match(correct_tweet):
        start_with_punctuation = re.findall(r'^[$]+\d+[.,;:?!\-*\/+]|^[$]+\d+[.,]+\d+[.,;:?!\-*\/+]|^\d+[$]+[.,;:?!\-*\/+]|^\d+[.,]+\d+[$]+[.,;:?!\-*\/+]|^[€]+\d+[.,;:?!\-*\/+]|^[€]+\d+[.,]+\d+[.,;:?!\-*\/+]|^\d+[€]+[.,;:?!\-*\/+]|^\d+[.,]+\d+[€]+[.,;:?!\-*\/+]', correct_tweet)
        word = re.findall(r'[$]+\d+[.,]+\d+|\d+[.,]+\d+[$]|[€]+\d+[.,]+\d+|\d+[.,]+\d+[€]|[$]+\d+|\d+[$]|[€]+\d+|\d+[€]', start_with_punctuation[0])
        correct_tweet = correct_tweet.replace(word[0],"money")
    return correct_tweet
    


# # Tempo
# 
# Função que remove referencias a tempo, substituindo pela palavra "time". Exemplo: "10:10 AM" passa a "time".

# In[15]:


def time_treatment(tweet):
    tweet = re.sub(r"(?<=\s)(1[0-2]|0?[1-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?(?=\s|$)|(?<=\s)(1[0-2]|0?[1-9]):([0-5]?[0-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?(?=\s|$)", "time", tweet)
    list_with_punctuation = [i.group(0) for i in re.finditer(r"(?<=\s)(1[0-2]|0?[1-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?[.,;?!\-*\/+](?=\s|$)|(?<=\s)(1[0-2]|0?[1-9]):([0-5]?[0-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?[.,;?!\-*\/+](?=\s|$)", tweet)]
    correct_tweet = tweet
    for word_punctuation in list_with_punctuation:
        punctuation = re.findall(r'[.,;?!\-*\/+]', word_punctuation)
        correct_tweet = correct_tweet.replace(word_punctuation,"time" + punctuation[0])
    if re.compile("^((1[0-2]|0?[1-9]):([0-5]?[0-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?)").match(correct_tweet):
        word = re.findall(r'^((1[0-2]|0?[1-9]):([0-5]?[0-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?)', correct_tweet)
        word = word[0][0].strip()
        correct_tweet = correct_tweet.replace(word,"time")
    if re.compile("^((1[0-2]|0?[1-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?)").match(correct_tweet):
        word = re.findall(r'^((1[0-2]|0?[1-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?)', correct_tweet)
        word = word[0][0].strip()
        correct_tweet = correct_tweet.replace(word,"time")
    if re.compile("^((1[0-2]|0?[1-9]):([0-5]?[0-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?[.,;?!\-*\/+])").match(correct_tweet):
        start_with_punctuation = re.findall(r'^((1[0-2]|0?[1-9]):([0-5]?[0-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?[.,;?!\-*\/+])', correct_tweet)
        word = re.findall(r'((1[0-2]|0?[1-9]):([0-5]?[0-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?)', start_with_punctuation[0][0])
        correct_tweet = correct_tweet.replace(word[0][0],"time")
    if re.compile("^((1[0-2]|0?[1-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?[.,;?!\-*\/+])").match(correct_tweet):
        start_with_punctuation = re.findall(r'^((1[0-2]|0?[1-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?[.,;?!\-*\/+])', correct_tweet)
        word = re.findall(r'((1[0-2]|0?[1-9]):([0-5]?[0-9])(\s*)(?i)(●?[AP]M)?)', start_with_punctuation[0][0])
        correct_tweet = correct_tweet.replace(word[0][0],"time")
    return correct_tweet


# # Sequência de caracteres repetidos
# 
# Função que converte uma palavra com mais de 2 caracteres iguais, para a mesma palavra repetida uma vez (sem os caracteres repetidos). Exemplo: "Golooo" passa a "Golo Golo".

# In[16]:


def remove_repetead_characters(word):
    return ''.join(c[0] for c in itertools.groupby(word))


def repetead_characters(tweet):
    correct_tweet = tweet
    list_matches = [i.group(0) for i in re.finditer(r"\S*([A-Za-z])(?i)\1{2,}\S*", tweet)]
    for word in list_matches:
        correct_tweet = correct_tweet.replace(word,remove_repetead_characters(word)+" "+remove_repetead_characters(word))
    return correct_tweet


# # Abreviaturas
# 
# Função que a partir de um ficherio de formato CSV, delimitado por ",", com o nome "abreviations.csv", converte as abreviações em palavras que cada abreviatura representa. Este ficheiro encontra-se na pasta "TM_Trabalho_1/Abbreviations".

# In[17]:


abbreviations = {}
with open('../TM_Trabalho_1/Abbreviations/abbreviations.csv', encoding="utf-8") as csvfile:
    for row in csvfile:
        row = row.replace("\n", "").replace('''"''', "").split(",", 1)
        key = row[0]
        value = row[1]
        abbreviations.update({key: value})
        


# In[18]:


def abbreviations_treatment(tweet):
    correct_tweet = []
    words = TweetTokenizer().tokenize(tweet)
    for word in words:
        if word in abbreviations.keys():
            correct_tweet.append(abbreviations.get(word))
        else:
            correct_tweet.append(word)
    return " ".join(correct_tweet)
    


# # Emoticons
# 
# Função que a partir de um ficherio de formato CSV, delimitado por ",", com o nome "emoticons.csv", converte os emoticons em palavras que cada emoticon representa. Este ficheiro encontra-se na pasta "TM_Trabalho_1/Emoticons".

# In[19]:


emoticons = {}
with open('../TM_Trabalho_1/Emoticons/emoticons.csv', encoding="ISO-8859-1") as csvfile:
    for row in csvfile:
        row = row.replace("\n", "").split(",", 1)
        key = row[0]
        value = row[1]
        emoticons.update({key: value})
        


# In[20]:


def emoticons_treatment(tweet):
    correct_tweet = []
    words = TweetTokenizer().tokenize(tweet)
    for word in words:
        if word in emoticons.keys():
            correct_tweet.append(emoticons.get(word))
        else:
            correct_tweet.append(word)
    return " ".join(correct_tweet)
    


# # Stop words
# 
# Função que remove os stop words.

# In[21]:


def stop_words_treatment(tweet):
    stop_words = set(stopwords.words('english'))
    words = TweetTokenizer().tokenize(tweet) 
    correct_tweet = [] 
    for word in words: 
        if word not in stop_words: 
            correct_tweet.append(word)
    return " ".join(correct_tweet)


# # Aplicação de um léxico de sentimentos
# 
# Nesta fase foi utilizado um classificador de sentimentos, baseado num léxico, neste caso foi utilizado o ficheiro NCR Word-Emotion Association Lexicon (EmoLex), em formato CSV.
# Deste ficheiro foram utilizados os campos "English", "Positive" e "Negative".
# Os dados sofreram um tratamento geral, referenciado nas funções anteriores.

# In[22]:


lexicon = {}
neutral = 0
positive = 0
negative = 0

csvfile = open("../TM/data/en/NCR-lexicon.csv", "r", encoding="utf-8")
csv_reader = csv.DictReader(csvfile, delimiter=";")

for row in csv_reader:
    lexicon.update({row["English"]: (row["Positive"], row["Negative"])})
    if (row["Positive"] == "1" and row["Negative"] == "1") or (row["Positive"] == "0" and row["Negative"] == "0"):
        neutral += 1
    if (row["Positive"] == "1" and row["Negative"] == "0"):
        positive += 1
    if (row["Positive"] == "0" and row["Negative"] == "1"):
        negative += 1

print("Numero de palavras neutras: {}".format(neutral))
print("Numero de palavras positivas: {}".format(positive))
print("Numero de palavras negativas: {}".format(negative))
print("Numero de palavras no lexico: {}".format(len(lexicon)))


# Função auxiliar que verifica quais os métodos a aplicar sem o tratamento da negação, neste caso se é aplicado simplesmente o léxico ou o Lemmatization com o léxico ou o Stemming com o léxico.
# Recebe os valores inseridos no método "run_lexicon" e aplica os métodos solicitados.
# Devolve tweet_pos_sentiment e tweet_neg_sentiment, que são 2 inteiros, que informam se o tweet tem tendência em ser positivo ou negativo.

# In[23]:


def sentimental_lexicon(tweet, with_wordNet, with_stemmer):
    tweet = data_treatment(tweet, False, False, False)
    words = TweetTokenizer().tokenize(tweet)
    frequency = FreqDist(words)
    tweet_pos_sentiment = 0
    tweet_neg_sentiment = 0
    
    for freq in frequency.most_common():
        sentimental_value = freq[1]
        lexicon_word = freq[0]
        
        if not with_wordNet and not with_stemmer:
            tweet_pos_sentiment, tweet_neg_sentiment = verify_lexicon_without_neg(lexicon_word, sentimental_value, tweet_pos_sentiment, tweet_neg_sentiment)
        if with_wordNet:
            lemmatizer = WordNetLemmatizer()
            lemmatizer_word = lemmatizer.lemmatize(lexicon_word)
            tweet_pos_sentiment, tweet_neg_sentiment = verify_lexicon_without_neg(lemmatizer_word, sentimental_value, tweet_pos_sentiment, tweet_neg_sentiment)
        if with_stemmer:
            stemmer = PorterStemmer()
            stemmer_word = stemmer.stem(lexicon_word)
            tweet_pos_sentiment, tweet_neg_sentiment = verify_lexicon_without_neg(stemmer_word, sentimental_value, tweet_pos_sentiment, tweet_neg_sentiment)
            
    return tweet_pos_sentiment, tweet_neg_sentiment


def verify_lexicon_without_neg(word, sentimental_value, tweet_pos_sentiment, tweet_neg_sentiment):
    if word in lexicon.keys():
        for lexicon_sentiment in lexicon.get(word):
            if lexicon.get(word).index(lexicon_sentiment) == 0:
                pos_lexicon_sentiment = lexicon_sentiment
            if lexicon.get(word).index(lexicon_sentiment) == 1:
                neg_lexicon_sentiment = lexicon_sentiment
                    
                if (pos_lexicon_sentiment == "1" and neg_lexicon_sentiment == "0") or (pos_lexicon_sentiment == "1" and neg_lexicon_sentiment == "1"):
                    tweet_pos_sentiment += sentimental_value
                if (pos_lexicon_sentiment == "0" and neg_lexicon_sentiment == "1") or (pos_lexicon_sentiment == "0" and neg_lexicon_sentiment == "0"):
                    tweet_neg_sentiment += sentimental_value
                    
    return tweet_pos_sentiment, tweet_neg_sentiment


# Função auxiliar que verifica quais os métodos a aplicar com o tratamento da negação, neste caso se é aplicado simplesmente o léxico ou o Lemmatization com o léxico ou o Stemming com o léxico. Recebe os valores inseridos no método "run_lexicon" e aplica os métodos solicitados. Devolve tweet_pos_sentiment e tweet_neg_sentiment, que são 2 inteiros, que informam se o tweet tem tendência em ser positivo ou negativo.

# In[24]:


def negation_treatment(tweet, with_wordNet, with_stemmer):
    tweet = data_treatment(tweet, True, False, False)
    words = TweetTokenizer().tokenize(tweet)
    words = util.mark_negation(words)
    frequency = FreqDist(words)
    tweet_pos_sentiment = 0
    tweet_neg_sentiment = 0
    
    for freq in frequency.most_common():
        sentimental_value = freq[1]
        lexicon_word = freq[0]
        
        if "_NEG" in lexicon_word:
            correct_word = lexicon_word.replace("_NEG","")
            correct_word = re.sub(r'[^\w\s]', "", correct_word)
            
            if not with_wordNet and not with_stemmer:
                tweet_neg_sentiment = verify_lexicon_with_neg(correct_word, sentimental_value, tweet_neg_sentiment)
            if with_wordNet:
                lemmatizer = WordNetLemmatizer()
                lemmatizer_word = lemmatizer.lemmatize(correct_word)
                tweet_neg_sentiment = verify_lexicon_with_neg(lemmatizer_word, sentimental_value, tweet_neg_sentiment)
            if with_stemmer:
                stemmer = PorterStemmer()
                stemmer_word = stemmer.stem(correct_word)
                tweet_neg_sentiment = verify_lexicon_with_neg(stemmer_word, sentimental_value, tweet_neg_sentiment)
        else:
            tweet_pos_sentiment += sentimental_value
            
    return tweet_pos_sentiment, tweet_neg_sentiment
            
            
def verify_lexicon_with_neg(word, sentimental_value, tweet_neg_sentiment):
    if word in lexicon.keys():
        tweet_neg_sentiment += sentimental_value
    return tweet_neg_sentiment


# Função principal do léxico que consoante o que for passado como argumento, calcula os valores para as métricas.
# Para o calculo destes valores, considera-se positivo se a variável tweet_pos_sentiment > tweet_neg_sentiment, e negativo se variável tweet_pos_sentiment <= tweet_neg_sentiment.

# In[25]:


def run_lexicon(with_wordNet, with_stemmer, with_negation_treatement):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    real_values = []
    pred_values = []
    pos_real = 0
    pos_pred = 0
    neg_real = 0
    neg_pred = 0
    
    for tweet in test_list:
        sentiment = tweet["class"]
        tweet = tweet["text"]
        
        if sentiment == 'pos': 
            real_values.append(1)
            pos_real += 1
        elif sentiment == 'neg': 
            real_values.append(0)
            neg_real += 1
            
        if not with_negation_treatement:
            tweet_pos_sentiment, tweet_neg_sentiment = sentimental_lexicon(tweet, with_wordNet, with_stemmer)
        else:
            tweet_pos_sentiment, tweet_neg_sentiment = negation_treatment(tweet, with_wordNet, with_stemmer)
        
        if sentiment == "pos" and (tweet_pos_sentiment > tweet_neg_sentiment):
            true_positive += 1
        if sentiment == "neg" and (tweet_pos_sentiment < tweet_neg_sentiment):
            true_negative += 1
        if not(sentiment == "pos" and (tweet_pos_sentiment > tweet_neg_sentiment)) and not(sentiment == "neg" and (tweet_pos_sentiment < tweet_neg_sentiment)):
            if tweet_pos_sentiment > tweet_neg_sentiment:
                false_positive += 1
            else:
                false_negative += 1  

        if tweet_pos_sentiment > tweet_neg_sentiment: 
            pred_values.append(1)
            pos_pred += 1
        if tweet_pos_sentiment <= tweet_neg_sentiment: 
            pred_values.append(0)
            neg_pred += 1

    metrics(true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred)
    
    
print("--- Léxico sem o tratamento da negação ---\n")
run_lexicon(False, False, False)

print("\n\n--- Léxico com WordNet ---\n")
run_lexicon(True, False, False)

print("\n\n--- Léxico com Stemming ---\n")
run_lexicon(False, True, False)

print("\n\n--- Léxico com o tratamento da negação ---\n")
run_lexicon(False, False, True)

print("\n\n--- Léxico com o tratamento da negação e com WordNet ---\n")
run_lexicon(True, False, True)

print("\n\n--- Léxico com o tratamento da negação e com Stemming ---\n")
run_lexicon(False, True, True)


# # Aprendizagem automática
# 
# Funções auxiliares de suporte aos algoritmos de Aprendizagem automática, com tratamento geral aplicado.
# Os métodos aplicados foram Lemmatization, Stemming, POS-tagging e tratamento da negação.
# Os algoritmos utilizados foram Naive Bayes, Logistic Regression e SVM.

# In[26]:


def get_newdocrep(texts, sentiments, text_sentiments):
    tokenizer = TweetTokenizer()
    docs = []

    for t in texts:
        doc = collections.Counter()
        for w in tokenizer.tokenize(t):
            doc[w] += 1
        docs.append(doc)
        
    voc_length = 3000

    tf = collections.Counter()
    df = collections.Counter()

    for d in docs:
        for w in d:
            tf[w] += d[w]
            df[w] += 1

    idfs = {}
    for w in tf:
        if tf[w] > 2:
            idfs[w] = np.log(len(docs)/df[w])

    voc = sorted(idfs, key=idfs.get, reverse=True)[:voc_length]
    
    indice = {}
    for i,w in enumerate(sorted(voc)):
        indice[w] = i
        
    docrep = []
    for d in docs:
        valores = np.zeros([len(voc)])
        for w in d:
            if w in indice:
                valores[ indice[w] ] = d[w]
        docrep.append ( valores )
        
    newdocrep = []
    for d,c in zip(docs, text_sentiments):
        docwords={}
        for w in d:
            if w in indice:
                docwords[w] = d[w]
        newdocrep.append ( (docwords, sentiments[c] ) )
    return newdocrep


def get_train_test_treatment(tweets, with_pos_tag, with_negation_treatment):
    tweets_treated = []
    for tweet in tweets:
        text = tweet["text"]
        sentiment = tweet["class"]
        
        if with_pos_tag:
            correct_tweet = data_treatment(text, True, True, with_pos_tag)
        else:
            if with_negation_treatment:
                correct_tweet = data_treatment(text, True, False, with_pos_tag)
            else:
                correct_tweet = data_treatment(text, False, False, with_pos_tag)
                
        tweet_treated = {'text': correct_tweet, 'class': sentiment}
        tweets_treated.append(tweet_treated)
    train_list, test_list = train_test_lists("Após Tratamento dos Tweets", tweets_treated, False)
    
    return train_list, test_list


def get_values_metrics(sentiment, pred_sentiment, true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred):
    if sentiment == "pos" and pred_sentiment == 'pos':
        true_positive += 1
    if sentiment == "neg" and pred_sentiment == 'neg':
        true_negative += 1
    if sentiment == "neg" and pred_sentiment == 'pos':
        false_positive += 1
    if sentiment == "pos" and pred_sentiment == 'neg':
        false_negative += 1
        
    if sentiment == 'pos': 
        real_values.append(1)
        pos_real += 1
    if sentiment == 'neg': 
        real_values.append(0)
        neg_real += 1
            
    if pred_sentiment == 'pos': 
        pred_values.append(1)
        pos_pred += 1
    if pred_sentiment == 'neg': 
        pred_values.append(0)
        neg_pred += 1
        
    return true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred


def get_classifies(train_list, test_list, with_wordNet, with_stemmer, with_pos, with_negation_treatement, classifier):
    sentiments_list = ["pos", "neg"]
    positive_sentiment = sentiments_list[0]
    negative_sentiment = sentiments_list[1]
    texts_list = []
    text_sentiments_list = []
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    real_values = []
    pred_values = []
    pos_real = 0
    pos_pred = 0
    neg_real = 0
    neg_pred = 0
    
    for tweet in train_list:
        text = tweet["text"]
        sentiment = tweet["class"]
        
        if with_wordNet or with_stemmer or with_pos:
            new_text = choose_options(text, with_wordNet, with_stemmer, with_pos, with_negation_treatement)
        
        if not with_wordNet and not with_stemmer and not with_pos:
            new_text = tweet["text"]
        
        texts_list.append(new_text)
        if sentiment == positive_sentiment:
            text_sentiments_list.append(0)
        if sentiment == negative_sentiment:
            text_sentiments_list.append(1)
            
    newdocrep = get_newdocrep(texts_list, sentiments_list, text_sentiments_list)
    
    if classifier == 'Naive Bayes':
        classifier = naivebayes.NaiveBayesClassifier.train(newdocrep)
    if classifier == 'Logistic Regression':
        classifier = maxent.MaxentClassifier.train(newdocrep, bernoulli=False, max_iter=1, trace=3)
    if classifier == 'SVM':
        classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(newdocrep)
    
    for tweet in test_list:
        text = tweet["text"]
        sentiment = tweet["class"]
        
        if with_wordNet or with_stemmer or with_pos:
            new_text = choose_options(text, with_wordNet, with_stemmer, with_pos, with_negation_treatement)
        
        if not with_wordNet and not with_stemmer and not with_pos:
            new_text = tweet["text"]
        
        words = TweetTokenizer().tokenize(new_text)
        doc = collections.Counter()
        for word in words:
            doc[word] += 1
    
        pred_sentiment = classifier.classify(doc)
        
        true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred = get_values_metrics(sentiment, pred_sentiment, true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred)
        
    return true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred
        

def choose_options(text, with_wordNet, with_stemmer, with_pos_tag, with_negation_treatement):
    if with_wordNet:
        lemmatizer = WordNetLemmatizer()
        lemmatizer_list = [lemmatizer.lemmatize(word) for word in text.split()]
        if with_negation_treatement:
            lemmatizer_list = util.mark_negation(lemmatizer_list)
            new_text = " ".join(lemmatizer_list)
        else:
            new_text = " ".join(lemmatizer_list)
            
    if with_stemmer:
        stemmer = PorterStemmer()
        stemmer_list = [stemmer.stem(word) for word in text.split()]
        if with_negation_treatement:
            stemmer_list = util.mark_negation(stemmer_list)
            new_text = " ".join(stemmer_list)
        else:
            new_text = " ".join(stemmer_list)
            
    if with_pos_tag:               
        if not with_wordNet and not with_stemmer:
            words = TweetTokenizer().tokenize(text)
            pos_tag_text = nltk.pos_tag(words)
            new_text = ""
        if with_wordNet or with_stemmer:
            words = TweetTokenizer().tokenize(new_text)
            pos_tag_text = nltk.pos_tag(words)
            new_text = ""
        for word_pos_tag in pos_tag_text:
            new_text += " " + word_pos_tag[0] + "_" + word_pos_tag[1]
    
    return new_text


# Função principal da Aprendizagem automática que consoante o que for passado como argumento, calcula os valores para as métricas.

# In[27]:


def run_classifier(train_list, test_list, with_treatment, with_wordNet, with_stemmer, with_pos_tag, with_negation_treatment, classifier):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    real_values = []
    pred_values = []
    pos_real = 0
    pos_pred = 0
    neg_real = 0
    neg_pred = 0
    
    if with_treatment:
        train_list, test_list = get_train_test_treatment(tweets, with_pos_tag, with_negation_treatment)
        
    true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred = get_classifies(train_list, test_list, with_wordNet, with_stemmer, with_pos_tag, with_negation_treatment, classifier)
            
    metrics(true_positive, true_negative, false_positive, false_negative, real_values, pred_values, pos_real, pos_pred, neg_real, neg_pred)


# # Naive Bayes

# In[28]:


print("***** NAIVE BAYES *****\n")

print("--- Sem features ---\n")
run_classifier(train_list, test_list, False, False, False, False, False, "Naive Bayes")

print("\n\n--- Com tratamento geral ---\n")
run_classifier(train_list, test_list, True, False, False, False, False, "Naive Bayes")

print("\n\n--- Com tratamento geral e WordNet ---\n")
run_classifier(train_list, test_list, True, True, False, False, False, "Naive Bayes")

print("\n\n--- Com tratamento geral e Stemming ---\n")
run_classifier(train_list, test_list, True, False, True, False, False, "Naive Bayes")

print("\n\n--- Com tratamento geral e POS-Tagging ---\n")
run_classifier(train_list, test_list, True, False, False, True, False, "Naive Bayes")

print("\n\n--- Com tratamento geral e tratamento da negação ---\n")
run_classifier(train_list, test_list, True, False, False, False, True, "Naive Bayes")

print("\n\n--- Com tratamento geral, WordNet e POS-Tagging ---\n")
run_classifier(train_list, test_list, True, True, False, True, False, "Naive Bayes")

print("\n\n--- Com tratamento geral, WordNet e tratamento da negação ---\n")
run_classifier(train_list, test_list, True, True, False, False, True, "Naive Bayes")

print("\n\n--- Com WordNet ---\n")
run_classifier(train_list, test_list, False, True, False, False, False, "Naive Bayes")

print("\n\n--- Com WordNet e POS-Tagging ---\n")
run_classifier(train_list, test_list, False, True, False, True, False, "Naive Bayes")

print("\n\n--- Com WordNet e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, True, False, False, True, "Naive Bayes")

print("\n\n--- Com Stemming ---\n")
run_classifier(train_list, test_list, False, False, True, False, False, "Naive Bayes")

print("\n\n--- Com Stemming e POS-Tagging ---\n")
run_classifier(train_list, test_list, False, False, True, True, False, "Naive Bayes")

print("\n\n--- Com Stemming e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, True, False, True, "Naive Bayes")

print("\n\n--- Com Stemming, POS-Tagging e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, True, True, True, "Naive Bayes")

print("\n\n--- Com POS-Tagging ---\n")
run_classifier(train_list, test_list, False, False, False, True, False, "Naive Bayes")

print("\n\n--- Com POS-Tagging e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, False, True, True, "Naive Bayes")

print("\n\n--- Com Tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, False, False, True, "Naive Bayes")


# # Logistic Regression

# In[29]:


print("***** LOGISTIC REGRESSION *****\n")

print("--- Sem features ---\n")
run_classifier(train_list, test_list, False, False, False, False, False, "Logistic Regression")

print("\n\n--- Com tratamento geral ---\n")
run_classifier(train_list, test_list, True, False, False, False, False, "Logistic Regression")

print("\n\n--- Com tratamento geral e WordNet ---\n")
run_classifier(train_list, test_list, True, True, False, False, False, "Logistic Regression")

print("\n\n--- Com tratamento geral e Stemming ---\n")
run_classifier(train_list, test_list, True, False, True, False, False, "Logistic Regression")

print("\n\n--- Com tratamento geral e POS-Tagging ---\n")
run_classifier(train_list, test_list, True, False, False, True, False, "Logistic Regression")

print("\n\n--- Com tratamento geral e tratamento da negação ---\n")
run_classifier(train_list, test_list, True, False, False, False, True, "Logistic Regression")

print("\n\n--- Com tratamento geral, WordNet e POS-Tagging ---\n")
run_classifier(train_list, test_list, True, True, False, True, False, "Logistic Regression")

print("\n\n--- Com tratamento geral, WordNet e tratamento da negação ---\n")
run_classifier(train_list, test_list, True, True, False, False, True, "Logistic Regression")

print("\n\n--- Com WordNet ---\n")
run_classifier(train_list, test_list, False, True, False, False, False, "Logistic Regression")

print("\n\n--- Com WordNet e POS-Tagging ---\n")
run_classifier(train_list, test_list, False, True, False, True, False, "Logistic Regression")

print("\n\n--- Com WordNet e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, True, False, False, True, "Logistic Regression")

print("\n\n--- Com Stemming ---\n")
run_classifier(train_list, test_list, False, False, True, False, False, "Logistic Regression")

print("\n\n--- Com Stemming e POS-Tagging ---\n")
run_classifier(train_list, test_list, False, False, True, True, False, "Logistic Regression")

print("\n\n--- Com Stemming e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, True, False, True, "Logistic Regression")

print("\n\n--- Com Stemming, POS-Tagging e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, True, True, True, "Logistic Regression")

print("\n\n--- Com POS-Tagging ---\n")
run_classifier(train_list, test_list, False, False, False, True, False, "Logistic Regression")

print("\n\n--- Com POS-Tagging e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, False, True, True, "Logistic Regression")

print("\n\n--- Com Tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, False, False, True, "Logistic Regression")


# # SVM

# In[30]:


print("***** SVM *****\n")

print("--- Sem features ---\n")
run_classifier(train_list, test_list, False, False, False, False, False, "SVM")

print("\n\n--- Com tratamento geral ---\n")
run_classifier(train_list, test_list, True, False, False, False, False, "SVM")

print("\n\n--- Com tratamento geral e WordNet ---\n")
run_classifier(train_list, test_list, True, True, False, False, False, "SVM")

print("\n\n--- Com tratamento geral e Stemming ---\n")
run_classifier(train_list, test_list, True, False, True, False, False, "SVM")

print("\n\n--- Com tratamento geral e POS-Tagging ---\n")
run_classifier(train_list, test_list, True, False, False, True, False, "SVM")

print("\n\n--- Com tratamento geral e tratamento da negação ---\n")
run_classifier(train_list, test_list, True, False, False, False, True, "SVM")

print("\n\n--- Com tratamento geral, WordNet e POS-Tagging ---\n")
run_classifier(train_list, test_list, True, True, False, True, False, "SVM")

print("\n\n--- Com tratamento geral, WordNet e tratamento da negação ---\n")
run_classifier(train_list, test_list, True, True, False, False, True, "SVM")

print("\n\n--- Com WordNet ---\n")
run_classifier(train_list, test_list, False, True, False, False, False, "SVM")

print("\n\n--- Com WordNet e POS-Tagging ---\n")
run_classifier(train_list, test_list, False, True, False, True, False, "SVM")

print("\n\n--- Com WordNet e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, True, False, False, True, "SVM")

print("\n\n--- Com Stemming ---\n")
run_classifier(train_list, test_list, False, False, True, False, False, "SVM")

print("\n\n--- Com Stemming e POS-Tagging ---\n")
run_classifier(train_list, test_list, False, False, True, True, False, "SVM")

print("\n\n--- Com Stemming e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, True, False, True, "SVM")

print("\n\n--- Com Stemming, POS-Tagging e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, True, True, True, "SVM")

print("\n\n--- Com POS-Tagging ---\n")
run_classifier(train_list, test_list, False, False, False, True, False, "SVM")

print("\n\n--- Com POS-Tagging e tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, False, True, True, "SVM")

print("\n\n--- Com Tratamento da negação ---\n")
run_classifier(train_list, test_list, False, False, False, False, True, "SVM")

