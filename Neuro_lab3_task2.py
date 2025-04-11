# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np


df = pd.read_csv('d:\Download\data.csv')
X = torch.Tensor(df.iloc[:, 0:3].values)
y = df.iloc[:, 4].values
y = torch.Tensor(np.where(y == "Iris-setosa", 1, -1).reshape(-1,1))

linear = nn.Linear(3, 1)
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.0001)


for i in range(0,5000):
    pred = linear(X)
    loss = lossFn(pred, y)
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())
    loss.backward()
    optimizer.step()