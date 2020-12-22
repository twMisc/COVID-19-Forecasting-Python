# %% [markdown]
# ## Goal: To predict the confirmed and deaths of COVID-19
# * We use the confirmed and deaths data from Johns Hopkins CSSE COVID-19 time series to train the model 
# * We generate the training data by the following process:
# %% [markdown]
#  ![Alt](./split_window.png)
# %% [markdown]
# * By deafult, the countries used for training are: US, Spain, Belgium, China, France, Germany, United Kingdom, Italy
# * Also, we leave the last 150 days to the test set
# %% [markdown]
# ### The following models are used to predict:
# 1. xgboost
# 2. linear regression
# 3. dense neural network with 2 hinden layers consists of 64 neurons
# %% [markdown]
# ## The predict score:
#  Let $\hat{Y} = \{\hat{\mathbb{y}_i}\}$ be the predicted values, and let $Y =\{ \mathbb{y}_i\}$ be the corresponding true values, the predict score is defined as  $$\text{score}  = 1 - \frac{\sum_i (\hat{\mathbb{y}_i} - \mathbb{y}_i)^2}{\sum_i (\mathbb{y}_i - \mu(Y))^2  },$$ where $\mu(Y)$ is the mean of $Y$.
# %%
from forecasting_setup import print_setups
print_setups()
# %%
from forecasting_multi import print_and_draw
# %%
print_and_draw('US')
# %%
print_and_draw('Spain')
# %%
print_and_draw('Belgium')
#%%
print_and_draw('China')
#%%
print_and_draw('France')
#%%
print_and_draw('Germany')
# %% 
print_and_draw('United Kingdom')
# %%
print_and_draw('Italy')
# %%
print_and_draw('Japan')
# %%
print_and_draw('India')
# %%
print_and_draw('Taiwan*')
# %%
print_and_draw('Indonesia')
# %%
print_and_draw('Malaysia')
# %%
print_and_draw('Korea, South')
# %%
print_and_draw('Vietnam')
# %%
print_and_draw('Russia')
# %%
