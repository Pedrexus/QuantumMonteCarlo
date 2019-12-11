import pandas as pd
from matplotlib import pyplot as plt

path = r'data.csv'
df = pd.read_csv(path, sep=";").set_index('alpha')

df['E'].plot()
plt.show()