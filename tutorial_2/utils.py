import matplotlib.pyplot as plt


def plot_line_graph(values, title="Line Graph", xlabel="X-axis", ylabel="Y-axis"):
    plt.plot(values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
