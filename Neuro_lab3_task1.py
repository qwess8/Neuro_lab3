# -*- coding: utf-8 -*-
import torch 
import random


x = torch.randint(1,4, (1,1),dtype=torch.int32)
print(x)

x = x.to(dtype=torch.float32)
x.requires_grad=True
print(x)

y = x ** 3
print(y)

z = y * random.randint(1,10)
print(z)

a = torch.exp(z)
print(a)

a.backward()
print(x.grad)