import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd

plt.xlabel("Batch")
plt.ylabel("Accuracy")

b_1, b_2, b_3, a_1, a_2, a_3 = [], [], [], [], [], []
av_1, av_2, av_3 = [0, 0, 0], [0, 0, 0], [0, 0, 0]       #平均准确率
num = [0, 0, 0, 0, 0, 0, 0, 0, 0]
batchs = [16, 32, 64]
file = pd.read_csv("result.csv", header=0, encoding = "utf-8")
raw = file.iloc[np.where(file["X-shot"].notnull())]
for index,row in raw.iterrows():
    shot = row['X-shot']
    batch = row['batch']
    accuracy = row['accuracy']
    if shot < 8:
        b_1.append(batch)
        a_1.append(accuracy)
        if batch < 20:
            av_1[0] += accuracy
            num[0] += 1
        elif batch < 40:
            av_1[1] += accuracy
            num[1] += 1
        else:
            av_1[2] += accuracy
            num[2] += 1
    elif shot < 16:
        b_2.append(batch)
        a_2.append(accuracy)
        if batch < 20:
            av_2[0] += accuracy
            num[3] += 1
        elif batch < 40:
            av_2[1] += accuracy
            num[4] += 1
        else:
            av_2[2] += accuracy
            num[5] += 1
    else:
        b_3.append(batch)
        a_3.append(accuracy)
        if batch < 20:
            av_3[0] += accuracy
            num[6] += 1
        elif batch < 40:
            av_3[1] += accuracy
            num[7] += 1
        else:
            av_3[2] += accuracy
            num[8] += 1
for i in range(3):
    av_1[i] = av_1[i] / num[i]
    av_2[i] = av_2[i] / num[i + 3]
    av_3[i] = av_3[i] / num[i + 6]


plt.xlim(8, 80)
x_major_locator = MultipleLocator(8)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)


plt.scatter(x = b_1, y = a_1, c = "blue", s = 24, alpha = 0.4, label = "5-shot")  #5-shot
plt.scatter(x = b_2, y = a_2, c = "red", s = 24, alpha = 0.4, label = "10-shot")   #10-shot
plt.scatter(x = b_3, y = a_3, c = "green", s = 24, alpha = 0.4, label = "20-shot") #20-shot
plt.legend() #显示标签

plt.plot(batchs, av_1, c = "blue", linewidth = 1, alpha = 0.8, linestyle = '--')
plt.plot(batchs, av_2, c = "red", linewidth = 1, alpha = 0.8, linestyle = '--')
plt.plot(batchs, av_3, c = "green", linewidth = 1, alpha = 0.8, linestyle = '--')

for i in range(3):
    plt.text(x = batchs[i], y = av_1[i] + 0.001, s = str('%.4f' % av_1[i]), c = "blue")
    plt.text(x = batchs[i], y = av_2[i] + 0.001, s = str('%.4f' % av_2[i]), c = "red")
    plt.text(x = batchs[i], y = av_3[i] + 0.001, s = str('%.4f' % av_3[i]), c = "green")
    
plt.title("ALBERT few-shot learning")
plt.show()