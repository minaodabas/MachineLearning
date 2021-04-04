#kutuphaneler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


carData = pd.read_csv('C:\\Users\\minao\\Desktop\\cars_dataset.csv')

x = carData.iloc[:,1:].values
y = carData.iloc[:,2:3].values


Transmission_TheCar = carData.iloc[:,3:4].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Transmission_TheCar[:,0] = le.fit_transform(carData.iloc[:,3:4])
ohe = preprocessing.OneHotEncoder()
Transmission_TheCar = ohe.fit_transform(Transmission_TheCar).toarray()
s1 = pd.DataFrame(data = Transmission_TheCar[:,0:2],index = range(72435),columns = ['Automatic','Manuel'])
s2 = pd.DataFrame(data = Transmission_TheCar[:,3],index = range(72435),columns = ['Semi-Auto'])
s_catTr =  pd.concat([s1,s2],axis=1) # Burada aracın manuel otomatik veya tam otoatik olma durumları ayarlandı.



##Araba Markalarının Sayısal verilere dönüştürülmesi
CarName =carData.iloc[:,9:].values
CarName[:,0] = le.fit_transform(carData.iloc[:,9:])
CarName = ohe.fit_transform(CarName).toarray()
s_CarName = pd.DataFrame(data = CarName,index = range(72435),columns = ['BMW','Ford',' Hyundai','Audi','Skoda','Toyota','VW'])




#Benzin tipinide makine öğrenmesinin algılayacağı şeylere çevireceğiz.
FuelType = carData.iloc[:,5:6].values
from sklearn import preprocessing
le_fuel = preprocessing.LabelEncoder()
FuelType[:,0] = le_fuel.fit_transform(carData.iloc[:,5:6])
ohe_fuel = preprocessing.OneHotEncoder()
FuelType = ohe_fuel.fit_transform(FuelType).toarray()

s1_fuel = pd.DataFrame(data=FuelType[:,0:1],index = range(72435),columns = ['Diesel'])
s2_fual = pd.DataFrame(data=FuelType[:,2:],index = range(72435),columns = ['Hybrid','Other','Petrol'])

s_FuelType = pd.concat([s1_fuel,s2_fual],axis=1)



## Burada ise fueltype ile vites tipini birleştireceğiz.
s_FuelAndcarTr = pd.concat([s_catTr,s_FuelType],axis=1)



## Benzin ve vites tipini tablodan ayırmak için bunu kullandık.
MillAge = carData.iloc[:,4:5].values
Tax_Mpg_EngineSize = carData.iloc[:,6:9].values



s_cardata2 = pd.DataFrame(data = MillAge,index = range(72435),columns = ['MillAge'])
s_cardata3 = pd.DataFrame(data = Tax_Mpg_EngineSize,index = range(72435),columns = ['Tax','Mpg','EngineSize'])


s_assembly = pd.concat([s_cardata2,s_cardata3],axis=1)



## ŞİMDİ VİTES VE FUELİ GENEL TABLO İLE BİRLESTİRECEGİZ
s_assembly2 = pd.concat([s_FuelAndcarTr,s_assembly],axis=1)


#Burada ise marka ve genel tabloyu birleştirerek son haline ulaştırdık.
s_total = pd.concat([s_assembly2,s_CarName],axis=1)



## Şimdi eğitim , buarada datalarımızı test ve train diye ayıracağız
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s_total,y,test_size=0.33,random_state=0)


# X_train ve X_test dosyaları makine öğrenmesinin uygulayabileceği sayılara dönüştürülür.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



# KNN UYGULAMASI
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(y_test)



cm = confusion_matrix( y_test, y_pred)
print(cm)
plt.show()
plt.scatter(y_test,y_pred)
#plt.plot(y_test,y_pred,color="blue")
