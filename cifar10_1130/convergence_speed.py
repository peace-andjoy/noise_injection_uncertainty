# %%
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd

# %%
root = './save/2024-12-02'
plt.figure(figsize=[16, 9], dpi=300)
for folder in os.listdir(root):
    df = pd.read_csv(os.path.join(root, folder, 'loss.txt'), sep='\s+')
    plt.plot(df['test'], label=folder)
    print(df['test'])
    print(df[df['test']<=0.9].index[0])
plt.legend()

# %%
