import numpy as np
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self, input_vectors, dest, init_weights, ni, gradient_diff):
        self.input_vectors = input_vectors
        self.dest = dest
        self.weights = init_weights
        self.ni = ni
        self.gradient_diff = gradient_diff
        self.weights_diff = np.ones([3,3])

    def dot_product(self, x, w):
        return sum([xi * wi for (xi, wi) in zip(x, w)])

    def activation_func(self, w, x):
        return 1/(1+np.exp(-self.dot_product(x, w)))

    def energy(self, d, y):
        return (d-y)**2

    def get_x2(self, x1):
        return [1, self.activation_func(self.weights[0], x1), self.activation_func(self.weights[1], x1)]

    def get_x3(self, x2):
        return self.activation_func(self.weights[2], x2)

    def gradient_matrix(self, d, output_vectors):
        [x3, x2, x1] = output_vectors
        e_deriv = 2*(x3-d)
        x3_deriv = e_deriv*x3*(1-x3)
        x22_deriv = x3_deriv*x2[2]*(1-x2[2])*self.weights[2, 2]
        x21_deriv = x3_deriv*x2[1]*(1-x2[1])*self.weights[2, 1]
        return np.array([[x21_deriv*x1[0], x21_deriv*x1[1], x21_deriv*x1[2]], [x22_deriv*x1[0], x22_deriv*x1[1], x22_deriv*x1[2]], [x3_deriv*x2[0], x3_deriv*x2[1], x3_deriv*x2[2]]], np.double)

    def get_updated_weights(self, energy_gradient):
        updated_weights = np.empty_like(self.weights)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                updated_weights[i, j] = self.weights[i, j] - self.ni*energy_gradient[i, j]
        return updated_weights

    def cycle(self, x1):
        x2 = self.get_x2(x1)
        x3 = self.get_x3(x2)
        return [x3, x2, [x1[0], x1[1], x1[2]]]

    def is_gradient_changing(self):
        for i in range(len(self.weights_diff)):
            for j in range(len(self.weights_diff[i])):
                if abs(self.weights_diff[i, j]) > self.gradient_diff:
                    return True
        return False

    def update_weights(self, energy_gradient):
        updated_weights = self.get_updated_weights(energy_gradient)
        self.weights_diff = updated_weights-self.weights
        self.weights = updated_weights

    def print_results(self, title, iteration, solution):
        print(title)
        print('Nr iteracji:', iteration)
        print('Rozwiązanie sieci:\n', solution)
        print('Końcowe wagi:\n', self.weights)

    def show_energy_diff_plots(energies_partial, energies_bulk):
        fig, (ax0, ax1) = plt.subplots(2, 1, constrained_layout=True, figsize=(9, 8))
        iter_p = range(len(energies_partial))
        ax0.set_title('Zmiana wartość energii w trybie energii cząstkowej')
        ax0.plot(iter_p, energies_partial[:, 0],label='Energia dla wektora [1,0,0]')
        ax0.plot(iter_p, energies_partial[:, 1],label='Energia dla wektora [1,0,1]')
        ax0.plot(iter_p, energies_partial[:, 2],label='Energia dla wektora [1,1,0]')
        ax0.plot(iter_p, energies_partial[:, 3],label='Energia dla wektora [1,1,1]')
        ax0.legend(loc='upper right')
        ax0.set_ylabel('Energia')
        ax0.set_xlabel('Nr iteracji')
        iter_b = range(len(energies_bulk))
        ax1.plot(iter_b, energies_bulk[:, 0],label='Energia dla wektora [1,0,0]')
        ax1.plot(iter_b, energies_bulk[:, 1],label='Energia dla wektora [1,0,1]')
        ax1.plot(iter_b, energies_bulk[:, 2],label='Energia dla wektora [1,1,0]')
        ax1.plot(iter_b, energies_bulk[:, 3],label='Energia dla wektora [1,1,1]')
        ax1.legend(loc='upper right')
        ax1.set_ylabel('Energia')
        ax1.set_xlabel('Nr iteracji')
        ax1.set_title('Zmiana wartość energii w trybie energii całkowitej')
        plt.show(block=True)

Network.show_energy_diff_plots = staticmethod(Network.show_energy_diff_plots)


class PartialEnergyNetwork(Network):
    def solve(self):
        solution = np.ones(4)
        energy = np.ones(4)
        iteration = 0
        energies = np.empty((0, 4), dtype=float)
        while self.is_gradient_changing() and iteration < 20000:
            for i in range(len(self.input_vectors)):
                output_vectors = self.cycle(self.input_vectors[i])
                energy_gradient = self.gradient_matrix(self.dest[i], output_vectors)
                energy[i] = self.energy(output_vectors[0], self.dest[i])
                solution[i] = output_vectors[0]
                self.update_weights(energy_gradient)
            iteration += 1
            energies = np.append(energies, np.array([energy]), axis=0)
        self.print_results('Tryb energii cząstkowej', iteration, solution)
        return energies


class BulkEnergyNetwork(Network):
    def solve(self):
        solution = np.ones(4)
        energy = np.ones(4)
        iteration = 0
        energies = np.empty((0, 4), dtype=float)
        while self.is_gradient_changing() and iteration < 20000:
            energy_gradient_sum = np.zeros([3, 3])
            for i in range(len(self.input_vectors)):
                output_vectors = self.cycle(self.input_vectors[i])
                energy_gradient_sum += self.gradient_matrix(self.dest[i], output_vectors)
                energy[i] = self.energy(output_vectors[0], self.dest[i])
                solution[i] = output_vectors[0]
            self.update_weights(energy_gradient_sum)
            iteration += 1
            energies = np.append(energies, np.array([energy]), axis=0)
        self.print_results('Tryb energii całkowitej', iteration, solution)
        return energies


def main():
    weights_init = np.array([[0.86, -0.16, 0.28], [0.82, -0.51, -0.89], [0.04, -0.43, 0.48]], np.double)
    x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], np.double)
    d = [0, 1, 1, 0]
    network_partial = PartialEnergyNetwork(x, d, weights_init, 0.5, 0.001)
    network_bulk = BulkEnergyNetwork(x, d, weights_init, 0.5, 0.0001)
    energies_pn = network_partial.solve()
    energies_bn = network_bulk.solve()
    Network.show_energy_diff_plots(energies_pn, energies_bn)

if __name__ == '__main__':
    main()
