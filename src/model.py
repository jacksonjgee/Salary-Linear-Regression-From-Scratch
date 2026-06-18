import pandas as pd
import matplotlib.pyplot as plt

PATH = "data/Salary_dataset.csv"

data = pd.read_csv(PATH)

def loss_function(m, c, data):
    total_error = 0
    n = len(data)

    for i in range(n):
        years_experience = data.YearsExperience.iloc[i]
        salary = data.Salary.iloc[i]

        prediction = m * years_experience + c
        total_error += (salary - prediction) ** 2

    return float(total_error / n)


c = 0
m = 121873.0 / 10.6

error = loss_function(m, c, data)
print("Mean squared error:", error)

def gradient_descent(m_now, c_now, data, L):
    m_gradient = 0
    c_gradient = 0
    n = len(data)

    for i in range(n):
        x = data.YearsExperience.iloc[i]
        y = data.Salary.iloc[i]

        m_gradient += -(2 / n) * x * (y - (m_now * x + c_now))
        c_gradient += -(2 / n) * (y - (m_now * x + c_now))

    m = m_now - m_gradient * L
    c = c_now - c_gradient * L
    return m, c

m = 0
c = 0
epochs = 100000
learning_rate = 0.0001

for i in range(epochs):
    m, c = gradient_descent(m, c, data, learning_rate)

    if i % 1000 == 0:
        error = loss_function(m, c, data)
        print(f"Epoch {i}: MSE = {error:.2f}")

print(m, c)
error = loss_function(m, c, data)
print("Mean squared error after training:", error)

plt.scatter(
    data.YearsExperience,
    data.Salary,
    color="black",
    label="Actual salaries"
)

x_line = [
    data.YearsExperience.min(),
    data.YearsExperience.max()
]

y_line = [
    m * x + c for x in x_line
]

plt.plot(
    x_line,
    y_line,
    color="red",
    label="Regression line"
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.close()