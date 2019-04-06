import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class Extrapolator:
    """Extrapolate and interpolate a sparse vector field into a dense one"""

    def __init__(self):
        return

    @staticmethod
    def plot_vector_field(x, y, step=1, scale=1, show_plot=True):
        assert x.shape == y.shape
        x_loc = np.arange(0, x.shape[1])
        y_loc = np.arange(0, x.shape[0])

        plt.quiver(x_loc[::step], y_loc[::step], x[::step, ::step], y[::step, ::step], angles='xy', scale_units='xy', scale=1*scale)

        if show_plot:
            plt.show()

    def extrapolate(self, x, y, z1, z2, out_size):
        # Given two sparse functions f1(x, y) = z1 and f2(x, y) = z2, calculate function values z1_out and z2_out on
        # each point of a regular grid with dimensions out_size

        # Fill result with zeros. You will need to change this.
        z1_out = np.zeros(out_size)
        z2_out = np.zeros(out_size)

        return z1_out, z2_out


if __name__ == '__main__':
    img_size = [100, 200]

    e = Extrapolator()

    # Generate a random sparse vector field
    n_samples = 60
    max_val = 7
    row = np.random.randint(img_size[0], size=[n_samples])
    col = np.random.randint(img_size[1], size=[n_samples])
    data_x = np.random.rand(n_samples) * max_val - (max_val / 2)
    data_y = np.random.rand(n_samples) * max_val - (max_val / 2)

    # Plot the sparse vector field
    x_orig = np.full(img_size, np.nan)
    y_orig = np.full(img_size, np.nan)
    x_orig[row, col] = data_x
    y_orig[row, col] = data_y
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title('Sparse vector field')
    e.plot_vector_field(x_orig, y_orig, show_plot=False)

    # Generate a dense vector field from the sparse one
    x, y = e.extrapolate(col, row, data_x, data_y, img_size)

    # Plot the dense vector field
    plt.subplot(1, 2, 2)
    plt.title('Dense vector field')
    e.plot_vector_field(x, y, step=5)

    # You might want to add more test cases with non-random data, to make sure everything works as expected
