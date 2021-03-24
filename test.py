import helper as hp
import pandas as pd
from constants import environment


train_data = pd.read_csv('data/test.csv')
res = hp.remove_excess(train_data)
print(res['insult'])
