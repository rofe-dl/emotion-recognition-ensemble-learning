import re
import nltk
nltk.download('punkt')

# Deals with cleaning text for now
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

def clean_text(data):
    data = "".join([word for word in data if word not in string.punctuation])
    data = word_tokenize(data)

    data = [word for word in data if word not in stopwords.words('english')]
    return data

t1 = 'i left with my bouquet of red and yellow tulips under my arm feeling slightly more optimistic than when i arrived'
t2 = 'i was feeling a little vain when i did this one'
t3 = 'i cant walk into a shop anywhere where i do not feel uncomfortable'

text = clean_text(t1)
text_train = clean_text(t2)
text_test = clean_text(t3)

print(text)
print(text_train)
print(text_test)
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts)
# sequence_train = tokenizer.texts_to_sequences(texts_train)
# sequence_test = tokenizer.texts_to_sequences(texts_test)
