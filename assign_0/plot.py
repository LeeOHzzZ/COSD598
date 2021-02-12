import matplotlib.pyplot as plt

def plot(data, x_label, y_label, fname):
    plt.plot(data[0], data[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(fname)
