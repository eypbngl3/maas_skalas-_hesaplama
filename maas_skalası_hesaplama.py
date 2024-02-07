#!/usr/bin/env python
# coding: utf-8

# # Polynomial Linear Regression
# Polynomial Linear Regression Genel Formülü :
# y = a+b1x +b2x^2+b3x^3+b4x^4+.....+bN*x^N

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Veri setimizi pandas yardımıyla alıp dataframe nesnemiz olan df'in içine aktarıyoruz.
df = pd.read_csv("C:\\Users\\pc\\Desktop\\Artificial İntelligence\\PROJELER\\maas_skalası_hesaplama\\polynomial.csv",sep=";")
df.head()  # Verinin ilk birkaç satırını görüntüle


# In[18]:


# bir adet polynomial regression nesnesi oluşturması için PolynomialFeatures fonksiyonunu çağırıyoruz.
# Bu fonksiyonu çağırırken polinomun derecesini (N) belitityoruz:
polynomial_regression = PolynomialFeatures(degree = 3)


# In[7]:


# Veri setimize bir bakalım
plt.scatter(df["deneyim"],df["maas"])
plt.xlabel("Deneyim(yıl)")
plt.ylabel("Maaş")
plt.savefig("1.png",dpi=300)
plt.show()
# Görüldüğü gibi doğrusal bir yapıda dağılmıyor veriler
# eğer biz bu veri setine linear regression uygularsak hiç uygun olmayan bir tahmin çizgisi görürüz:


# In[9]:


reg = LinearRegression()
reg.fit(df["deneyim"].values.reshape(-1, 1), df["maas"])

plt.xlabel("Deneyim (yıl)")
plt.ylabel("Maaş")

plt.scatter(df["deneyim"], df["maas"])

x_ekseni = df["deneyim"]
y_ekseni = reg.predict(df["deneyim"].values.reshape(-1, 1))
plt.plot(x_ekseni, y_ekseni, color="green", label="linear regression")
plt.legend()
plt.show()


# # Bu veri seti için regression çeşitlerinden polynomial regression uygulanması gerektiğine karar verdik. Şimdi nasıl uyguladığımıza bakalım:
# x değerimizi polinom yukarıdaki fonksiyonuna uyacak şekilde uyarlanmasını sağlıyoruz.
# Yani => 1,x,x^2(N=2) şeklinde

# In[25]:


polynomial_regression = PolynomialFeatures(degree=4)
x_polynomial = polynomial_regression.fit_transform(df[["deneyim"]])  # "deneyim" sütununu 2. dereceden polinom şekline dönüştür


# In[26]:


# regression model nesnemizi olan reg nesnemizi oluşturup bunun fit metodunu çağırarak x_polynomial ve y eksenlerini fit ediyoruz.
# yani regresyon modelimizi mevcut gerçek veirlerle eğitiyoruz :
reg = LinearRegression()
reg.fit(x_polynomial,df["maas"])


# # Artık modelimiz hazır ve eğitilmiş , şimdi eldeki verilere göre modelimiz nasıl bir sonuç grafiği oluşturuyor onu görelim :
# 

# In[27]:


y_head = reg.predict(x_polynomial)
plt.plot(df["deneyim"],y_head,color ="red",label="polynomial regression")
plt.legend()

# veri setimizi de noktalı olarak scatter edelim.
plt.scatter(df["deneyim"],df["maas"])

plt.show()


# Gördüğümüz gibi kesinlikle uymuş diyeibiliriz, polynomial regression doğru bir seçim.Şimdi bir de N=3  veya 4 yapıp görelim polinom derecesini arttırdığımızda daha güzeli fit edecek mi acaba ?

# In[28]:


x_polynomial1=polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial1)


# # Alacağı maaş çok güzel bir şekilde şirket politikasına fit etmiş oluyor.
