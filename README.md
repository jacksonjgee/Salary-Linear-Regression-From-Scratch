# Linear Regression from Scratch

A simple Python project that implements linear regression from scratch using gradient descent.

The model learns the relationship between years of experience and salary without using Scikit-learn.

## Overview

This project includes:

* Mean Squared Error as the loss function
* Gradient descent
* Manual calculation of the slope and intercept
* A visualisation of the fitted regression line

The model uses:

Learning rate = 0.0001

Epochs = 10000

## Result

<video src="assets/linear_regression_training.mp4" controls width="700">
</video>

## Project Structure

```text
.
├── data/
│   └── Salary_dataset.csv
├── images/
│   └── Figure_1.png
├── src/
│   └── model.py
├── main.py
├── requirements.txt
└── README.md
```

## Running the Project

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python main.py
```

## Technologies

* Python
* Pandas
* Matplotlib
