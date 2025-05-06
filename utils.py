import matplotlib.pyplot as plt


def plot_line_graph(*values_lists, title="Line Graph", xlabel="X-axis", ylabel="Y-axis", labels=None):
    for i, values in enumerate(values_lists):
        if labels:
            plt.plot(values, label=labels[i])
        else:
            plt.plot(values)  # no label if not provided

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if labels:
        plt.legend()

    plt.show()
