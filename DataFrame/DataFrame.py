import pandas as pd

# 1. Створення DataFrame та конвертація OrderDate у datetime
data = {
    'OrderID': [1001, 1002, 1003],
    'Customer': ['Alice', 'Bob', 'Alice'],
    'Product': ['Laptop', 'Chair', 'Mouse'],
    'Category': ['Electronics', 'Furniture', 'Electronics'],
    'Quantity': [1, 2, 3],
    'Price': [1500, 180, 25],
    'OrderDate': ['2023-06-01', '2023-06-03', '2023-06-05']
}

df = pd.DataFrame(data)
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
print("1. DataFrame з конвертованою датою:")
print(df)
print("\n")

# 2. Додавання стовпця TotalAmount
df['TotalAmount'] = df['Quantity'] * df['Price']
print("2. DataFrame з новим стовпцем TotalAmount:")
print(df)
print("\n")

# 3. Виведення статистики
print("3a. Сумарний дохід магазину:", df['TotalAmount'].sum())
print("3b. Середнє значення TotalAmount:", df['TotalAmount'].mean())
print("3c. Кількість замовлень по кожному клієнту:")
print(df['Customer'].value_counts())
print("\n")

# 4. Замовлення з сумою покупки > 500
print("4. Замовлення з сумою покупки > 500:")
print(df[df['TotalAmount'] > 500])
print("\n")

# 5. Сортування за OrderDate у зворотному порядку
df_sorted = df.sort_values('OrderDate', ascending=False)
print("5. Відсортована таблиця за датою:")
print(df_sorted)
print("\n")

# 6. Замовлення між 5 і 10 червня включно
start_date = pd.to_datetime('2023-06-05')
end_date = pd.to_datetime('2023-06-10')
print("6. Замовлення з 5 по 10 червня:")
print(df[(df['OrderDate'] >= start_date) & (df['OrderDate'] <= end_date)])
print("\n")

# 7. Групування за категорією
grouped = df.groupby('Category').agg({
    'Quantity': 'sum',
    'TotalAmount': 'sum'
}).rename(columns={
    'Quantity': 'Кількість товарів',
    'TotalAmount': 'Загальна сума продажів'
})
print("7. Групування за категорією:")
print(grouped)
print("\n")

# 8. ТОП-3 клієнтів за сумою покупок
top_customers = df.groupby('Customer')['TotalAmount'].sum().sort_values(ascending=False).head(3)
print("8. ТОП-3 клієнтів за загальною сумою покупок:")
print(top_customers)