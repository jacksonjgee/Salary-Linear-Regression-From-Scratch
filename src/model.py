import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def run():
    PATH = "data/Salary_dataset.csv"

    data = pd.read_csv(PATH)

    # Create assets folder if it does not already exist
    os.makedirs("assets", exist_ok=True)

    def loss_function(m, c, data):
        total_error = 0
        n = len(data)

        for i in range(n):
            x = data.YearsExperience.iloc[i]
            y = data.Salary.iloc[i]

            prediction = m * x + c
            total_error += (y - prediction) ** 2

        return float(total_error / n)

    def gradient_descent(m_now, c_now, data, learning_rate):
        m_gradient = 0
        c_gradient = 0
        n = len(data)

        for i in range(n):
            x = data.YearsExperience.iloc[i]
            y = data.Salary.iloc[i]

            prediction = m_now * x + c_now

            m_gradient += -(2 / n) * x * (y - prediction)
            c_gradient += -(2 / n) * (y - prediction)

        m = m_now - m_gradient * learning_rate
        c = c_now - c_gradient * learning_rate

        return m, c

    # Initial parameters
    m = 0
    c = 80000

    epochs = 100000
    learning_rate = 0.0001

    # Stores snapshots for animation and MSE graph
    history = []
    mse_history = []

    for epoch in range(epochs):
        m, c = gradient_descent(m, c, data, learning_rate)

        if epoch % 1000 == 0:
            mse = loss_function(m, c, data)

            history.append((epoch, m, c, mse))
            mse_history.append((epoch, mse))

            print(f"Epoch {epoch}: MSE = {mse:.2f}")

    final_mse = loss_function(m, c, data)

    print("Final m:", m)
    print("Final c:", c)
    print("Final MSE:", final_mse)

    x = data.YearsExperience
    y = data.Salary

    x_line = [
        data.YearsExperience.min(),
        data.YearsExperience.max()
    ]

    y_line = [
        m * x_value + c for x_value in x_line
    ]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        x,
        y,
        color="black",
        label="Actual salaries"
    )

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

    plt.savefig("assets/final_regression_line.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    epochs_list = [item[0] for item in mse_history]
    mse_values = [item[1] for item in mse_history]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs_list,
        mse_values,
        marker="o",
        label="MSE"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE vs Epochs")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig("assets/mse_vs_epochs.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        x,
        y,
        color="black",
        label="Actual salaries"
    )

    animated_line, = ax.plot(
        [],
        [],
        color="red",
        label="Regression line"
    )

    text = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        verticalalignment="top"
    )

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Linear Regression Fitting the Data")
    ax.legend()
    ax.grid(alpha=0.3)

    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(y.min() - 10000, y.max() + 10000)

    def update(frame):
        epoch, current_m, current_c, mse = history[frame]

        current_y_line = [
            current_m * x_value + current_c for x_value in x_line
        ]

        animated_line.set_data(x_line, current_y_line)

        text.set_text(
            f"Epoch: {epoch}\n"
            f"MSE: {mse:,.2f}\n"
            f"m: {current_m:,.2f}\n"
            f"c: {current_c:,.2f}"
        )

        return animated_line, text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=100,
        blit=True
    )

    writer = FFMpegWriter(fps=10, bitrate=1800)

    animation.save(
        "assets/linear_regression_training.mp4",
        writer=writer
    )

    plt.close()