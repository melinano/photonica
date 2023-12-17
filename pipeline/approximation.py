import numpy as np
import matplotlib.pyplot as plt

def read_and_process_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    lines = data.strip().split("\n")
    x = [int(line.split("\t")[0]) for line in lines]
    y = [float(line.split("\t")[1]) for line in lines]
    return np.array(x), np.array(y)

def format_equation(coefficients):
    terms = []
    degree = len(coefficients) - 1

    for i, coef in enumerate(coefficients):
        if degree - i == 1:
            term = f"{coef:+.5f}x"
        elif degree - i == 0:
            term = f"{coef:+.5f}"
        else:
            term = f"{coef:+.5f}x^{degree - i}"

        terms.append(term)

    equation = " ".join(terms)
    return equation


def relative_error(y_pred, y):
    error = np.abs(y_pred - y)
    return np.sum(error) / (len(error) * (np.max(y) - np.min(y))) * 100


def get_curve_parameters(y):
    depth = max(y) - min(y)
    modification_diameter = len(y)
    return depth, modification_diameter


def fit_polyfit(data_path, degree=5):
    x, y = read_and_process_data(data_path)

    coefficients = np.polyfit(x, y, deg=degree)

    y_pred = np.polyval(coefficients, x)

    error = relative_error(y_pred, y)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b.', label='Original Data')
    ax.plot(x, y_pred, 'r-', label='Symbolic Regression')
    ax.legend()

    depth, diameter_mod = get_curve_parameters(y_pred)
    return coefficients, error, fig, depth, diameter_mod
