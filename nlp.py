import pandas as pd
import csv

# Yorumları saklamak için bir liste oluşturuyoruz
yorumlar = []

# CSV dosyasını okuyup verileri yorumlar listesine eklemek için gerekli kod
with open('Restaurant_Reviews.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, quotechar='"', delimiter=',', quoting=csv.QUOTE_MINIMAL)
    
    # Başlık satırını atlıyoruz
    next(reader)

    for row in reader:
        # Satırın en az 2 sütun içerip içermediğini kontrol edin
        if len(row) < 2:
            continue  # Eğer satır eksikse atla

        metin = row[0]  # Metin kolonu
        
        # Liked sütunu sayısal olup olmadığını kontrol ediyoruz
        liked = int(row[1]) if row[1].isdigit() else None
        
        # Her bir yorumu bir tuple olarak yorumlar listesine ekliyoruz
        yorumlar.append((metin, liked))

# Yorumları DataFrame'e dönüştürme
yorumlar_df = pd.DataFrame(yorumlar, columns=['Metin', 'Liked'])

import re


import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

stopwords = nltk.download('stopwords')

from nltk.corpus import stopwords

derlem = []
for i in range(988):    
    yorum = re.sub('[^a-zA-z]',' ',yorumlar_df['Metin'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = yorumlar_df.iloc[:,1].values # bağımlı değişken


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,)



from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
















