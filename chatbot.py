import nltk 
import numpy as np 
import random 
import string
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity


f=open('C:/Users/Erenn/Desktop/deneme.txt','r',errors = 'ignore') 
raw=f.read() 
raw=raw.lower()# küçük harfe dönüştürür 
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('omw-1.4')
sent_tokens = nltk.sent_tokenize(raw)# cümle listesine dönüştürür 
word_tokens = nltk.word_tokenize(raw)# kelimelerin listesine dönüştürür

lemmer = nltk.stem.WordNetLemmatizer() #WordNet, NLTK'da yer alan anlamsal odaklı bir İngilizce sözlüktür. 
def LemTokens(tokens): return [lemmer.lemmatize(token) for token in tokens] 
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation) 
def LemNormalize(text): return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("Merhaba", "Hello", "Hi", "Selam","hey") 
GREETING_RESPONSES = ["Merhaba Ben Robo", "Mrb Ben Robo", "Hello I'm Robo", "Selam Ben Robo"] 
def greeting(sentence): 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS: 
            return random.choice(GREETING_RESPONSES)

def response(user_response): 
    robo_response='' 
    sent_tokens.append(user_response) 
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') 
    tfidf = TfidfVec.fit_transform(sent_tokens) 
    vals = cosine_similarity(tfidf[-1], tfidf) 
    idx=vals.argsort()[0][-2] 
    flat = vals.flatten() 
    flat.sort() 
    req_tfidf = flat[-2] 
    if(req_tfidf==0): 
        robo_response=robo_response+"Üzgünüm Seni Anlayamadım Tekrar Sorarmısın..." 
        return robo_response 
    else: 
        robo_response = robo_response+sent_tokens[idx] 
        return robo_response

flag=True 
print("ROBO: Benim adım Robo. ChatBots hakkındaki sorularınızı cevaplayacağım. Çıkmak istiyorsanız Görüşmek Üzere... ") 
while(flag==True):
    user_response = input() 
    user_response=user_response.lower() 
    if(user_response!='gorusuruz'): 
        if(user_response=='tesekkurler' or user_response=='sagol' ): 
            flag=False 
            print("ROBO: Rica Ederim..") 
        else: 
            if(greeting(user_response)!=None): 
                print("ROBO: "+greeting(user_response)) 
            else: 
                print("ROBO: ",end="") 
                print(response(user_response)) 
                sent_tokens.remove(user_response) 
    else: 
        flag=False 
        print("ROBO: Görüşmek Üzere")

