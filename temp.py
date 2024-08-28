# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# # Veriyi yükleme
# veriler = pd.read_csv('odev_tenis.csv')

# # Kategorik verileri sayısal değerlere dönüştürme
# le = LabelEncoder()
# veriler['windy'] = le.fit_transform(veriler['windy'])
# veriler['play'] = le.fit_transform(veriler['play'])

# ohe = OneHotEncoder()
# outlook = ohe.fit_transform(veriler[['outlook']]).toarray()
# outlookDF = pd.DataFrame(data=outlook, index=range(14), columns=['Bulutlu','Yagmurlu','Gunesli'])

# # Gerekli sütunları seçme ve birleştirme
# selected_columns = veriler[['humidity', 'temperature', 'play', 'windy']]
# sonveriler = pd.concat([outlookDF, selected_columns], axis=1)

# # Bağımlı değişkeni belirleme (humidity)
# humidity = sonveriler['humidity'].values

# def backward_elimination(X, y, significance_level=0.05):
#     num_vars = len(X[0])
#     temp = np.zeros((X.shape[0], X.shape[1]))
#     for i in range(0, num_vars):
#         regressor_OLS = sm.OLS(y, X).fit()
#         max_var = max(regressor_OLS.pvalues).astype(float)
#         adjR_before = regressor_OLS.rsquared_adj
#         if max_var > significance_level:
#             for j in range(0, num_vars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == max_var):
#                     temp[:, j] = X[:, j]
#                     X = np.delete(X, j, 1)
#                     tmp_regressor = sm.OLS(y, X).fit()
#                     adjR_after = tmp_regressor.rsquared_adj
#                     if adjR_before >= adjR_after:
#                         X_rollback = np.hstack((X, temp[:, [0, j]]))
#                         X_rollback = np.delete(X_rollback, j, 1)
#                         return X_rollback
#                     else:
#                         continue
#     regressor_OLS.summary()
#     return X
# # Bağımsız değişken matrisini hazırlama
# X = np.append(arr=np.ones((14, 1)).astype(int), values=sonveriler.iloc[:, 1:], axis=1)
# X = np.array(X, dtype=float)

# # Backward Elimination uygulama
# X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]  # İlk olarak tüm bağımsız değişkenleri içerir
# X_modeled = backward_elimination(X_opt, humidity)

# # Sonuçları gösterme
# model = sm.OLS(humidity, X_modeled).fit()
# print(model.summary())

# # Eğitim ve test verilerini ayırma
# X_train, X_test, y_train, y_test = train_test_split(X_modeled, humidity, test_size=0.33, random_state=0)

# # Linear Regression modelini eğitme
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# # Test seti üzerinde tahmin yapma
# y_pred = regressor.predict(X_test)
# print(y_pred)

# # Backward Elimination'ı tekrar tekrar uygulayarak gereksiz değişkenleri çıkarma
# X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]
# X_modeled = backward_elimination(X_opt, humidity)

# # Sonuçları gösterme
# model = sm.OLS(humidity, X_modeled).fit()
# print(model.summary())

# # Eğitim ve test verilerini ayırma
# X_train, X_test, y_train, y_test = train_test_split(X_modeled, humidity, test_size=0.33, random_state=0)

# # Linear Regression modelini eğitme
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# # Test seti üzerinde tahmin yapma
# y_pred = regressor.predict(X_test)
# print(y_pred)



#######################

# POLİNOMAL REGRESYON



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


x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]

X = x.values
Y = y.values
print(veriler.corr(numeric_only=True))

lin_reg = LinearRegression()

lin_reg.fit(X, Y)



model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
# Polinomal regresyon

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)

 #♠ Lineer regresyon değerine polynomial olarak aldığımız x değerlerini vererek polinom regresyonu yapıyoruz.
 # lin_reg2 aslında poly_reg olarak yazılabilir.
lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)

#♣ tahminler


print("Poly OLS")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())
# Polinomal regresyon

# Support Vector Regression

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

print("SVR OLS")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
# Polinomal regresyon


# Karar Ağacı

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)



print("DT OLS")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())
# Polinomal regresyon
# Random Forest



from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=0,n_estimators=10)

rf_reg.fit(X,Y.ravel())





print("Random Forest Tahmini : ",rf_reg.predict(X))




print("Random Forest R2 Değeri")

print(r2_score(Y,rf_reg.predict(X)))

print("RF OLS")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())

