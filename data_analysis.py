import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset
dataset = load_iris()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data["species"] = pd.Categorical.from_codes(dataset.target, dataset.target_names)

# Exploration
print(data.head())
print(data.info())
print(data.isnull().sum())

# Basic stats
print(data.describe())
print(data.groupby("species").mean())

# 1. Line chart
data["cum_sepal_length"] = data["sepal length (cm)"].cumsum()
data["cum_sepal_length"].plot(title="Cumulative Sepal Length over Samples")
plt.xlabel("Sample")
plt.ylabel("Cumulative Sepal Length")
plt.savefig("line_chart.png")
plt.clf()

# 2. Bar chart
data.groupby("species")["petal length (cm)"].mean().plot(kind="bar", title="Avg Petal Length by Species")
plt.ylabel("Petal Length (cm)")
plt.savefig("bar_chart.png")
plt.clf()

# 3. Histogram
data["sepal length (cm)"].plot(kind="hist", bins=10, title="Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.savefig("histogram.png")
plt.clf()

# 4. Scatter plot
sns.scatterplot(data=data, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Sepal vs Petal Length")
plt.savefig("scatter.png")
plt.clf()

print("\nObservations:")
print("- Setosa has the shortest petals, Virginica the longest.")
print("- Sepal length increases roughly with petal length across species.")
