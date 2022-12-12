import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("compiled_data/compiled_pokemon_data.csv")

types = ["normal",
        "fighting",
        "flying",
        "poison",
        "ground",
        "rock",
        "bug",
        "ghost",
        "steel",
        "fire",
        "water",
        "grass",
        "electric",
        "psychic",
        "ice",
        "dragon",
        "dark",
        "fairy"]

count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for _, row in df.iterrows():
    label = row['types']
    label = label[1:-1]
    label = label.split(',')
    label = [int(x) for x in label]
    count = np.add(count, label)

plt.figure(figsize=(20,10))
plt.title("Histograma dos tipos de Pokemon")
plt.ylabel("Quantidade")
plt.xlabel("Tipo")
sns.barplot(x=types, y=count)
plt.savefig(f'Histograma.png', bbox_inches='tight')
plt.show()
