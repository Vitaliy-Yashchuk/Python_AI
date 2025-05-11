import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
data = pd.DataFrame({
    'Experience': np.random.randint(0, 6, 200),
    'Grade': np.round(np.random.uniform(7, 12, 200), 1),
    'EnglishLevel': np.random.randint(1, 6, 200),
    'Age': np.random.randint(18, 30, 200),
    'EntryTestScore': np.random.randint(400, 1000, 200)
})
data['Accepted'] = (
    (data['Experience'] > 2) &
    (data['Grade'] > 9) &
    (data['EnglishLevel'] >= 3) &
    (data['EntryTestScore'] > 650)
).astype(int)

X = data[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y = data['Accepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
english_range = np.arange(1, 6)
test_score_range = np.linspace(400, 1000, 100)

grid_english, grid_test_score = np.meshgrid(english_range, test_score_range)
Z = []

for i in range(grid_english.shape[0]):
    row = []
    for j in range(grid_english.shape[1]):
        sample = pd.DataFrame([{
            'Experience': 3,  # фіксовані значення
            'Grade': 10,
            'EnglishLevel': grid_english[i, j],
            'Age': 22,
            'EntryTestScore': grid_test_score[i, j]
        }])
        prob = model.predict_proba(sample)[0][1]
        row.append(prob)
    Z.append(row)

Z = np.array(Z)

plt.figure(figsize=(10, 6))
cp = plt.contourf(grid_english, grid_test_score, Z, cmap='coolwarm', levels=20)
plt.colorbar(cp, label='Probability of Acceptance')
plt.xlabel('English Level')
plt.ylabel('Entry Test Score')
plt.title('Probability of Acceptance vs English Level and Entry Test Score')
plt.show()
