import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv("./data/65002.csv")
print(data)


# Saat sütunundaki boşlukları kaldırın

tarih = data.iloc[:100,0:1].values

saat = data.iloc[:100,1:2].values
basınc = data.iloc[:100,2:3].values

print(tarih)
print(saat)
print(basınc)


plt.figure(figsize=(10, 6)) 


plt.plot(tarih,basınc, linestyle='-')

plt.xlabel('Saat (Time)')
plt.ylabel('Basınç (Pressure)')
plt.title('Pressure vs Time')
plt.show()