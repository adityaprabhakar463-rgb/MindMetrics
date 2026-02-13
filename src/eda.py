import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/study_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape: ")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

corr = df.corr()

plt.figure()
sns.heatmap(corr, annot = True)
plt.title("Correlation Matrix")
plt.show()