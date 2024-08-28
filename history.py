# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Tue May 28 16:38:46 2024)---
debugfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Wed May 29 11:53:09 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Wed Jun 26 22:16:24 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Thu Jun 27 08:30:58 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')
runcell(0, 'C:/Users/Doğukan/.spyder-py3/temp.py')
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Thu Jun 27 13:55:06 2024)---
runcell(0, 'C:/Users/Doğukan/.spyder-py3/temp.py')
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')
runcell(0, 'C:/Users/Doğukan/.spyder-py3/temp.py')
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)


# Yas kolonundaki boş değerleri ortalama ile doldurma
# veriler['yas'].fillna(veriler['yas'].mean(), inplace=True)

#encoder: Kategorik -> Numeric
windy = veriler.iloc[:,3:4].values
print(windy)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,0])

print(windy)

outlook = veriler.iloc[:,0:1].values
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)
outlook = veriler.iloc[:,0:1].values
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)
windy = veriler.iloc[:,3:4].values
print(windy)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,0])

print(windy)
windy = veriler.iloc[:,3:4].values
print(windy)
print(windy)
windy[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(windy)
windy[:,-2] = le.fit_transform(veriler.iloc[:,-2])

print(windy)
le = preprocessing.LabelEncoder()

windy[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(windy)
runcell(0, 'C:/Users/Doğukan/.spyder-py3/temp.py')
print(windy)
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)


# Yas kolonundaki boş değerleri ortalama ile doldurma
# veriler['yas'].fillna(veriler['yas'].mean(), inplace=True)

#encoder: Kategorik -> Numeric
windy = veriler.iloc[:,3:4].values
print(windy)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,-1] = le.fit_transform(windy)

print(windy)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,-1] = le.fit_transform(windy.iloc[:,0:1].values)

print(windy)
windy = veriler.iloc[:,3:4].values
print(windy)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,0])

print(windy)
le = preprocessing.LabelEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,3:4])

print(windy)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,0])

print(windy)
windy = veriler.iloc[:,3:4].values
print(windy)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,0])

print(windy)
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')
windy = veriler.iloc[:,3:4].values
print(windy)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,3])

print(windy)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,3])

print(windy)
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')
runcell(0, 'C:/Users/Doğukan/.spyder-py3/temp.py')
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# #2.veri onisleme
# #2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# #2.veri onisleme
# #2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Fri Jun 28 18:32:43 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Fri Jun 28 21:28:27 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Tue Jul  9 17:59:53 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,:1]
X = x.values
Y = y.values






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]
X = x.values
Y = y.values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]
X = x.values
Y = y.values

# lin_reg = LinearRegression()

# lin_reg.fit(X, Y)

# plt.scatter(X,Y,color="black")
# plt.plot(X,lin_reg.predict(X))
# plt.show()

# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
 
 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

plt.scatter(X,Y)
plt.plot(x,lin_reg2.predict(x_poly)) #♦ polynomyal gösterim
plt.plot(X,lin_reg.predict(X)) # lineer gösterim
plt.show()

#♣ tahminler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]
X = x.values
Y = y.values
x.drop('unvan',axis=1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]
X = x.values
Y = y.values
x = x.drop('unvan',axis=1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]
X = x.values
Y = y.values
x = x.drop('unvan',axis=1)
x = x.drop('kidem',axis=1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]
X = x.values
Y = y.values
x = x.drop('unvan',axis=1)
x = x.drop('Kidem',axis=1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]
X = x.values
Y = y.values
x = x.drop('unvan',axis=1)
x = x.drop('Kidem',axis=1)

# lin_reg = LinearRegression()

# lin_reg.fit(X, Y)

# plt.scatter(X,Y,color="black")
# plt.plot(X,lin_reg.predict(X))
# plt.show()

# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
 
 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

plt.scatter(X,Y)
plt.plot(x,lin_reg2.predict(x_poly)) #♦ polynomyal gösterim
plt.plot(X,lin_reg.predict(X)) # lineer gösterim
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Kidem',axis=1)
X = x.values
Y = y.values
# lin_reg = LinearRegression()

# lin_reg.fit(X, Y)

# plt.scatter(X,Y,color="black")
# plt.plot(X,lin_reg.predict(X))
# plt.show()

# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
 
 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

plt.scatter(X,Y)
plt.plot(x,lin_reg2.predict(x_poly)) #♦ polynomyal gösterim
plt.plot(X,lin_reg.predict(X)) # lineer gösterim
plt.show()

x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Kidem',axis=1)
X = x.values
Y = y.values
# lin_reg = LinearRegression()

# lin_reg.fit(X, Y)

# plt.scatter(X,Y,color="black")
# plt.plot(X,lin_reg.predict(X))
# plt.show()

# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
 
 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

plt.scatter(X[:,0],Y)
plt.plot(x,lin_reg2.predict(x_poly)) #♦ polynomyal gösterim
plt.plot(X,lin_reg.predict(X)) # lineer gösterim
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Kidem',axis=1)
X = x.values
Y = y.values
# lin_reg = LinearRegression()

# lin_reg.fit(X, Y)

# plt.scatter(X,Y,color="black")
# plt.plot(X,lin_reg.predict(X))
# plt.show()

# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
 
 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

plt.scatter(X[:,0],Y)
plt.plot(x,lin_reg2.predict(x_poly)) #♦ polynomyal gösterim
plt.plot(X,lin_reg.predict(X)) # lineer gösterim
plt.show()

x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Kidem',axis=1)
X = x.values
Y = y.values
# lin_reg = LinearRegression()

# lin_reg.fit(X, Y)

# plt.scatter(X,Y,color="black")
# plt.plot(X,lin_reg.predict(X))
# plt.show()

# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
 
 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

plt.scatter(X[:,0],Y)
plt.plot(x,lin_reg2.predict(x_poly)) #♦ polynomyal gösterim
plt.show()
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[10,100]])))

x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('calisanID',axis=1)
X = x.values
Y = y.values

x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Calisan ID',axis=1)
X = x.values
Y = y.values


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Calisan ID',axis=1)
X = x.values
Y = y.values
# lin_reg = LinearRegression()

# lin_reg.fit(X, Y)

# plt.scatter(X,Y,color="black")
# plt.plot(X,lin_reg.predict(X))
# plt.show()

# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
 
 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

plt.scatter(X[:,0],Y)
plt.plot(x,lin_reg2.predict(x_poly)) #♦ polynomyal gösterim

plt.show()

#♣ tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Calisan ID',axis=1)
X = x.values
Y = y.values
lin_reg = LinearRegression()

lin_reg.fit(X, Y)

plt.scatter(X,Y,color="black")
plt.plot(X,lin_reg.predict(X))
plt.show()

# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
 
 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

plt.scatter(X[:,0],Y)
plt.plot(x,lin_reg2.predict(x_poly)) #♦ polynomyal gösterim

plt.show()
x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Calisan ID',axis=1)
X = x.values
Y = y.values
lin_reg = LinearRegression()

lin_reg.fit(X, Y)

# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
 
 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

plt.scatter(X[:,0],Y)
plt.plot(x,lin_reg2.predict(x_poly)) #♦ polynomyal gösterim

plt.show()

## ---(Wed Jul 10 09:58:40 2024)---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Calisan ID',axis=1)
X = x.values
Y = y.values

lin_reg = LinearRegression()

lin_reg.fit(X, Y)

plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(X)) #♦ polynomyal gösterim
plt.show()


model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
# Polinomal regresyon
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]

x = x.drop('unvan',axis=1)
x = x.drop('Calisan ID',axis=1)
X = x.values
Y = y.values

lin_reg = LinearRegression()

lin_reg.fit(X, Y)



model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
# Polinomal regresyon
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
veriler = pd.read_csv("maaslar_yeni.csv")


#  Ünvanları Encode edelim / Videoda Yapılmıyor o yüzden geçiyorum.

# le = LabelEncoder()
# unvanlar = veriler.iloc[:,0:1].values

# unvanlar = le.fit_transform(unvanlar[:,0])

# Lineer regresyon nasıldı ?


x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:]

X = x.values
Y = y.values

lin_reg = LinearRegression()

lin_reg.fit(X, Y)



model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
%clear
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')
%clear
debugfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Wed Jul 10 17:44:19 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/classification.py', wdir='C:/Users/Doğukan/.spyder-py3')
runfile('C:/Users/Doğukan/.spyder-py3/temp.py', wdir='C:/Users/Doğukan/.spyder-py3')
runfile('C:/Users/Doğukan/.spyder-py3/classification.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Sun Jul 14 09:58:01 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/classification.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Sun Jul 14 16:22:27 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/classification.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Sun Jul 14 22:49:48 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/classification.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Sun Jul 14 23:23:37 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/classification.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Wed Aug 21 15:04:06 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/classification.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Thu Aug 22 12:02:48 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/classification.py', wdir='C:/Users/Doğukan/.spyder-py3')
runcell(0, 'C:/Users/Doğukan/.spyder-py3/classification.py')
runfile('C:/Users/Doğukan/.spyder-py3/classification.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Mon Aug 26 11:02:04 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/nlp.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Tue Aug 27 13:36:43 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/nlp.py', wdir='C:/Users/Doğukan/.spyder-py3')
runfile('C:/Users/Doğukan/.spyder-py3/untitled0.py', wdir='C:/Users/Doğukan/.spyder-py3')
runfile('C:/Users/Doğukan/.spyder-py3/keras.py', wdir='C:/Users/Doğukan/.spyder-py3')
runfile('C:/Users/Doğukan/.spyder-py3/xgboost.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Tue Aug 27 17:24:44 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/xgboost.py', wdir='C:/Users/Doğukan/.spyder-py3')
runfile('C:/Users/Doğukan/.spyder-py3/xgboost_test.py', wdir='C:/Users/Doğukan/.spyder-py3')

## ---(Wed Aug 28 14:42:28 2024)---
runfile('C:/Users/Doğukan/.spyder-py3/cnn.py', wdir='C:/Users/Doğukan/.spyder-py3')