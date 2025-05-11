import matplotlib.pyplot as plt
import numpy as np

#task_1
x = np.linspace(-10, 10, 500)
y = np.sin(x)

plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.title("Графік функції sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()

#task_2
data = np.random.normal(loc=5, scale=2, size=1000)

plt.figure(figsize=(6, 4))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Гістограма нормального розподілу (μ=5, σ=2)")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.grid(True)
plt.show()

#task_3
labels = ['Геймдев', 'Фітнес', 'Малювання', 'Настільні ігри', 'Подорожі']
sizes = [30, 25, 15, 20, 10]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Мої хобі")
plt.axis('equal')
plt.show()

#task_4
fruits = ['Яблуко', 'Банан', 'Апельсин', 'Груша']
fruit_data = [
    np.random.normal(loc=150, scale=15, size=100),  # Яблука
    np.random.normal(loc=120, scale=10, size=100),  # Банани
    np.random.normal(loc=130, scale=12, size=100),  # Апельсини
    np.random.normal(loc=140, scale=14, size=100)   # Груші
]

plt.figure(figsize=(8, 5))
plt.boxplot(fruit_data, labels=fruits)
plt.title("Box-plot: Маса фруктів")
plt.ylabel("Маса (грам)")
plt.grid(True)
plt.show()

#task_5
x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, color='green', alpha=0.6)
plt.title("Точкова діаграма з рівномірного розподілу")
plt.xlabel("X (0–1)")
plt.ylabel("Y (0–1)")
plt.grid(True)
plt.show()
