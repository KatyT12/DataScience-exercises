import IPython.display
from torch import nn
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
import Interrupt



url = "https://www.cl.cam.ac.uk/teaching/current/DataSci/data/xkcd.csv"
xkcd = pandas.read_csv(url)

class LogPr(nn.Module):

    def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(1.0))
            self.b = torch.nn.Parameter(torch.tensor(1.0))
            self.c = torch.nn.Parameter(torch.tensor(0.0))
            self.σ = torch.nn.Parameter(torch.tensor(1.0))

    def μ(self, x):
        return self.a + self.b*x + self.c*x**2

    def forward(self, y, x):
        σ2 = self.σ**2
        return torch.sum(-0.5 * torch.log(2*np.pi*σ2) - torch.pow(y - self.μ(x), 2) / (2 * σ2))

x = torch.tensor(xkcd.x, dtype=torch.float)
y = torch.tensor(xkcd.y, dtype=torch.float)
logPr = LogPr()
#print(logPr(y,x))


#Optimizing in pytorch

optimizer = torch.optim.Adam(logPr.parameters())
"""
for epoch in range(10000): # Number of gradient descent steps
    optimizer.zero_grad() # Reset gradients of all optimized tensore s
    loglik = logPr(y,x)
    (-loglik).backward() # Compute gradiant, backwards because we want maximum
    optimizer.step()

print(logPr(y,x))
"""
# Nested modules

class QuadraticCurve(nn.Module):

    def __init__(self):
        super.__init__()
        self.a = nn.parameter(torch.tensor(1.0))
        self.b = nn.parameter(torch.tensor(1.0))
        self.c = nn.parameter(torch.tensor(0.0))

    def forward(self,x):
        return self.a + self.b * x + self.c * x** 2

class RQuadratic(nn.Module):

        def __init__(self):
            super.__init__()
            self.μ = QuadraticCurve()
            self.σ = nn.parameter(torch.tensor(1.0))

        def forward(self,y,x):
            σ2 = self.σ**2
            return torch.sum( -0.5 * torch.log(σ2 * 2 * torch.pi) - torch.pow(y - self.μ(x),2)/(2 * σ2))

#########################


# Getting answers out of pytorch

"""
fig, ax = plt.subplots(figsize=(4,3))
xnew = np.linspace(0,10,100)
ynew = 0
with torch.no_grad():
    ynew = logPr.μ(xnew) # Mean

ynew.detach().numpy()
σhat = logPr.σ.item()

ax.plot(xnew, ynew)
ax.scatter(xkcd.x, xkcd.y, marker="x")
ax.fill_between(xnew, ynew - 2  * σhat,ynew + 2*σhat, color = 'steelblue', alpha=0.6)
plt.show()
"""

# Making it interactive

"""
When we optimizae a function using gradient descent how many iterations should we use? It's hard to know
when we're just starting out with a new model and we have no experiece, and w're still experimenting.
I like to run my optimization sinteractively, in an infinite loop, showing the fit every few iterations. Every so often I interrupt,
explore the plots in more detail and then resume.

I've written a piece of magic Python code ot help with this, a class called Interruptable. Use it as follows. In Jupyiter
we can interrupt the while loop using menu option Kernel interrupt or by a keyboard shortcut ESC. Can resume by rerunning the cell
with Interruptable()
"""


def plot_quad(model):
    fig, ax = plt.subplots(figsize=(4, 3))
    xnew = torch.linspace(0, 10, 100)
    with torch.no_grad():
        ynew = model.μ(xnew)  # Mean

    ynew.detach().numpy()
    xnew.detach().numpy()
    σhat = model.σ.item()

    ax.plot(xnew, ynew)
    ax.scatter(xkcd.x, xkcd.y, marker="x")
    ax.fill_between(xnew, ynew - 2 * σhat, ynew + 2 * σhat, color='steelblue', alpha=0.6)
    plt.show()

epoch = 0

# making it interactive
"""
with Interrupt.Interruptable() as check_interrupted:
    while True:
        check_interrupted()
        optimizer.zero_grad()

        loglik = logPr(y,x)
        (-loglik).backward()
        optimizer.step()
        epoch += 1
        if epoch % 10 == 0:
            IPython.display.clear_output(wait=True)
            print(f'epoch={epoch} loglik={loglik.item():.4} σ={logPr.σ.item():.4}')
            plot_quad(logPr)
"""

class RWiggle(nn.Module):
    def __init__(self):
        super().__init__()
        # self.μ maps R^(n×1) to R^(n×1)
        self.μ = nn.Sequential(
            nn.Linear(1, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1)
        )
        self.σ = nn.Parameter(torch.tensor(1.0))

    def logPr(self,y,x):
        # x and y are tensors of shap (n,)
        # Reshape x to be (n,1) applu μ, then drop the last dimension
        m = self.μ(x[:,None])[:,0]
        σ2 = self.σ**2
        return torch.sum(- 0.5*torch.log(2*np.pi*σ2) - torch.pow(y - m, 2) / (2*σ2))

def plot_wiggle(model):
    with torch.no_grad():
        xnew = torch.linspace(0,10,100)[:,None]
        ynew = model.μ(xnew)
        xnew = xnew.detach().numpy()[:,0]
        ynew = ynew.detach().numpy()[:, 0]
        σ = model.σ.item()

    fig, ax = plt.subplots()
    ax.set_ylim([0, 12])
    ax.fill_between(xnew, ynew-2*σ, ynew + 2*σ, color='steelblue', alpha=0.6)
    ax.plot(xnew, ynew, color = 'steelblue')
    ax.scatter(x,y,color ='black', marker='+', alpha=.8)
    plt.show()
"""
model = RWiggle()
x = torch.tensor(xkcd.x, dtype=torch.float)
y = torch.tensor(xkcd.y, dtype=torch.float)
optimizer = torch.optim.Adam(model.parameters())
with Interrupt.Interruptable() as check_interrupted:
    while True:
        check_interrupted()
        optimizer.zero_grad()

        loglik = model.logPr(y,x)
        (-loglik).backward()
        optimizer.step()
        epoch += 1
        if epoch % 200 == 0:
            IPython.display.clear_output(wait=True)
            print(f'epoch={epoch} loglik={loglik.item():.4} σ={model.σ.item():.4}')
            plot_wiggle(model)
"""
# Batched gradient descent

data  = torch.tensor(np.column_stack([xkcd.x,xkcd.y]), dtype=torch.float)
data_batched = torch.utils.data.DataLoader(data,batch_size=5,shuffle=True)

# Modify code to use batched data descent


model = RWiggle()
optimizer = torch.optim.Adam(model.parameters())
epoch = 0

with Interrupt.Interruptable() as check_interrupted:
    while True:
        check_interrupted()
        for b in data_batched:
            optimizer.zero_grad()
            loglik = torch.sum(model.logPr(b[:,1,None], b[:,0,None]))
            (-loglik).backward()
            optimizer.step()
        epoch += 1
        if epoch % 200 == 0:
            IPython.display.clear_output(wait=True)
            print(f'epoch={epoch} loglik={loglik.item():.4} σ={model.σ.item():.4}')
            plot_wiggle(model)











