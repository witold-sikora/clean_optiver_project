```python
data_dir = './data/train_data.csv'
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv(data_dir, index_col=0)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stock_id</th>
      <th>time_id</th>
      <th>alpha_time</th>
      <th>beta_time</th>
      <th>book_kurt</th>
      <th>book_skew</th>
      <th>log_volume</th>
      <th>mean_logtime</th>
      <th>order_mean</th>
      <th>order_std</th>
      <th>...</th>
      <th>shift_ratio</th>
      <th>shift_std</th>
      <th>sigma</th>
      <th>sigma_mean</th>
      <th>spread_kurt</th>
      <th>spread_mean</th>
      <th>spread_skew</th>
      <th>spread_std</th>
      <th>trade_impact</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>0.050380</td>
      <td>0.174265</td>
      <td>0.007629</td>
      <td>0.026514</td>
      <td>0.035063</td>
      <td>0.943416</td>
      <td>0.022148</td>
      <td>0.015094</td>
      <td>...</td>
      <td>0.593180</td>
      <td>0.006898</td>
      <td>0.004499</td>
      <td>0.065248</td>
      <td>0.010011</td>
      <td>0.066576</td>
      <td>0.026382</td>
      <td>0.031956</td>
      <td>0.421919</td>
      <td>0.004136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>11</td>
      <td>0.096677</td>
      <td>0.845970</td>
      <td>0.007387</td>
      <td>0.026936</td>
      <td>0.102071</td>
      <td>0.982827</td>
      <td>0.007390</td>
      <td>0.007865</td>
      <td>...</td>
      <td>0.812863</td>
      <td>0.008493</td>
      <td>0.001204</td>
      <td>0.011702</td>
      <td>0.010885</td>
      <td>0.020666</td>
      <td>0.080775</td>
      <td>0.022965</td>
      <td>0.376186</td>
      <td>0.001445</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>16</td>
      <td>0.989200</td>
      <td>0.985355</td>
      <td>0.006981</td>
      <td>0.023311</td>
      <td>0.039368</td>
      <td>0.994824</td>
      <td>0.021316</td>
      <td>0.013535</td>
      <td>...</td>
      <td>0.991944</td>
      <td>0.007234</td>
      <td>0.002369</td>
      <td>0.012486</td>
      <td>0.010236</td>
      <td>0.042966</td>
      <td>0.019015</td>
      <td>0.015643</td>
      <td>0.354362</td>
      <td>0.002168</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>31</td>
      <td>0.796320</td>
      <td>0.531626</td>
      <td>0.007096</td>
      <td>0.031630</td>
      <td>0.018559</td>
      <td>0.990192</td>
      <td>0.095781</td>
      <td>0.041700</td>
      <td>...</td>
      <td>0.965184</td>
      <td>0.007062</td>
      <td>0.002574</td>
      <td>0.015367</td>
      <td>0.010438</td>
      <td>0.060936</td>
      <td>0.043633</td>
      <td>0.060858</td>
      <td>0.089499</td>
      <td>0.002195</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>62</td>
      <td>0.940945</td>
      <td>0.968397</td>
      <td>0.013284</td>
      <td>0.057840</td>
      <td>0.307853</td>
      <td>0.986450</td>
      <td>0.109210</td>
      <td>0.043204</td>
      <td>...</td>
      <td>0.690482</td>
      <td>0.014763</td>
      <td>0.001894</td>
      <td>0.010211</td>
      <td>0.008786</td>
      <td>0.021896</td>
      <td>0.028323</td>
      <td>0.017671</td>
      <td>0.155979</td>
      <td>0.001747</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
y = df.target
x = df.drop("target", axis=1)
```


```python
df = pd.read_csv(data_dir, index_col='Unnamed: 0')

df = df[['stock_id', 'time_id'] + 
        sorted([c for c in df.columns if c not in ['stock_id', 'time_id', 'target']]) + 
        ['target']]



df.to_csv(data_dir)
```


```python
fig, ax = plt.subplots(figsize=(8, 6))

hexbin = ax.hexbin(df["sigma"], df["target"], 
                   gridsize=100,
                   cmap='twilight',
                   mincnt=1)

ax.set_title('Sigma vs Target (Hexbin Density)', fontsize=14, fontweight='bold')
ax.set_xlabel('Sigma', fontsize=12)
ax.set_ylabel('Target', fontsize=12)
plt.colorbar(hexbin, ax=ax, label='Count per bin')
plt.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(8, 6))

hexbin = ax.hexbin(df["spread_mean"], df["target"], 
                   gridsize=100,
                   cmap='twilight',
                   mincnt=1)

ax.set_title('Spread Mean vs Target (Hexbin Density)', fontsize=14, fontweight='bold')
ax.set_xlabel('Spread Mean', fontsize=12)
ax.set_ylabel('Target', fontsize=12)
plt.colorbar(hexbin, ax=ax, label='Count per bin')
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(df.select_dtypes(include=[np.number]).corr(), 
            annot=False, 
            fmt='.2f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            ax=ax)

plt.tight_layout()
plt.show()
```


    
![png](plotting_files/plotting_6_0.png)
    



    
![png](plotting_files/plotting_6_1.png)
    



    
![png](plotting_files/plotting_6_2.png)
    



```python
import matplotlib.pyplot as plt
import seaborn as sns

sigma_by_stock = df.groupby("stock_id")["sigma"].mean()
target_by_stock = df.groupby("stock_id")["target"].mean()

fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=False)

sigma_by_stock.plot.hist(bins=25, 
                         density=True,
                         alpha=0.6, 
                         color='steelblue',
                         edgecolor='black',
                         linewidth=0.5,
                         ax=axes[0])

sns.kdeplot(data=sigma_by_stock, 
            color='darkred', 
            linewidth=2,
            ax=axes[0])

axes[0].set_title('Distribution of Average Sigma by Stock', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Average Sigma', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].legend(['Histogram', 'KDE Density'])

target_by_stock.plot.hist(bins=25, 
                          density=True,
                          alpha=0.6, 
                          color='forestgreen',
                          edgecolor='black',
                          linewidth=0.5,
                          ax=axes[1])

sns.kdeplot(data=target_by_stock, 
            color='darkorange', 
            linewidth=2,
            ax=axes[1])

axes[1].set_title('Distribution of Average Target by Stock', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Average Target', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].legend(['Histogram', 'KDE Density'])

plt.tight_layout()
plt.show()
```


    
![png](plotting_files/plotting_7_0.png)
    

