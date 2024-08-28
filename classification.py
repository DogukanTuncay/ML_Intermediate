
import pandas as pd

#2.veri onisleme
# #2.1.veri yukleme
# veriler = pd.read_excel('iris.xlsx',engine='openpyxl')
# #pd.read_csv("veriler.csv")
# #test


# x = veriler.iloc[:,0:4].values #bağımsız değişkenler
# y = veriler.iloc[:,4:].values #bağımlı değişken


# #verilerin egitim ve test icin bolunmesi
# from sklearn.model_selection import train_test_split

# x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

# #verilerin olceklenmesi
# from sklearn.preprocessing import StandardScaler

# sc=StandardScaler()

# X_train = sc.fit_transform(x_train)
# X_test = sc.transform(x_test)


# from sklearn.linear_model import LogisticRegression
# logr = LogisticRegression(random_state=0)
# logr.fit(X_train,y_train)

# y_pred = logr.predict(X_test)



# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test,y_pred)
# print(cm)



# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train,y_train)

# y_pred = knn.predict(X_test)
# print("KNN")
# cm = confusion_matrix(y_test,y_pred)
# print(cm)



# from sklearn.svm import SVC
# svc = SVC(kernel='rbf')
# svc.fit(X_train,y_train)

# y_pred = svc.predict(X_test)

# cm = confusion_matrix(y_test,y_pred)
# print('SVC')
# print(cm)



# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)

# y_pred = gnb.predict(X_test)

# cm = confusion_matrix(y_test,y_pred)
# print('GNB')
# print(cm)


# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier(criterion = 'entropy')

# dtc.fit(X_train,y_train)
# y_pred = dtc.predict(X_test)

# cm = confusion_matrix(y_test,y_pred)
# print('DTC')
# print(cm)


# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=100, criterion = 'entropy')
# rfc.fit(X_train,y_train)

# y_pred = rfc.predict(X_test)
# cm = confusion_matrix(y_test,y_pred)
# print('RFC')
# print(cm)


    
# # 7. ROC , TPR, FPR değerleri 

# y_proba = rfc.predict_proba(X_test)
# print(y_proba[:,0])

# from sklearn import metrics
# fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
# print(fpr)
# print(tpr)



#HC

import matplotlib.pyplot as  plt
from sklearn.cluster import AgglomerativeClustering
veriler = pd.read_csv("musteriler.csv")
X = veriler.iloc[:,3:].values


ac = AgglomerativeClustering(n_clusters=4)

Y_tahmin = ac.fit_predict(X)

print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100,c="red") 
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,c="blue")
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,c="green")
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100,c="gray")
plt.show()

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X,method="ward"))

plt.show()


veri = pd.read_csv("Ads_CTR_Optimisation.csv")

# import random
# # Random Selection
# N = 10000
# d = 10
# toplam = 0
# secilenler = []
# for n in range(0,N):
#     ad = random.randrange(d)
#     secilenler.append(ad)
#     odul = veri.values[n,ad]
#     toplam = toplam + odul
    
# plt.hist(secilenler)

# plt.show()

# UCB
import math
N = 10000 # 10.000 satır veri var.
d = 10 # ilan sayısı
toplam=0 # toplam ödül
oduller = [0]*d # ilk başta bütün ilanların ödülü 0
tiklamalar = [0]*d
secilenler = []
for n in range(0,N):
    ad = 0 # seçilen ilan
    max_ucb = 0
    for i in range(0,d):
        if(tiklamalar[i] > 0):  
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2* math.log(n) / tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb=N*10
        if max_ucb < ucb: # maxtan büyük ise güncellensin
            max_ucb = ucb
            ad = i
            
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] +1
    odul = veri.values[n,ad]
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul


print('Toplam Odul :')
print(toplam)
plt.hist(secilenler)

plt.show()






import random


N = 10000 # 10.000 satır veri var.
d = 10 # ilan sayısı
toplam=0 # toplam ödül
secilenler = []
birler = [0]*d
sifirlar = [0]*d

for n in range(0,N):
    ad = 0 # seçilen ilan
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate(birler[i] +1 , sifirlar[i] +1)
        if(rasbeta > max_th):
            max_th = rasbeta
            ad = i
            
            
    secilenler.append(ad)
    
    odul = veri.values[n,ad]
    if(odul == 1):
        birler[ad]+=1
    else:
        sifirlar[ad]+=1
    toplam = toplam + odul


print('Toplam Odul :')
print(toplam)
plt.hist(secilenler)

plt.show()


































