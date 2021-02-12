import matplotlib.pyplot as plt

def plot(data, x_label, y_label, fname):
    plt.plot(data[0], linestyle='-', marker='+', label='training')
    plt.plot(data[1], linestyle='--', marker='.', label='testing')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylim((min(data[0]+data[1]), max(data[0]+data[1])))
    plt.legend()
    plt.title(fname)
    plt.savefig(fname)
