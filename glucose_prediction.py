from argparse import ArgumentParser
from csv import reader
from math import exp, log
from operator import itemgetter
from plotly import graph_objs as go
from plotly.offline import plot
from random import seed, random, randint, randrange

import datetime
import numpy as np
import os


class KohonenNetwork(object):

    def __init__(self, rows, columns):
        self.computional_layer = NodesLattice(rows, columns)

    def train(self, training_set, epochs, initial_learning_rate):
        map_radius = self.computional_layer.count_radius()  # sigma_zero
        time_constant = epochs / log(map_radius)  # lambda

        for iteration in range(epochs):
            random_input = training_set[randint(0, len(training_set) - 1)]
            bmu = self.get_best_matching_unit(random_input)
            decay_function = exp(-iteration / time_constant)
            learning_rate = initial_learning_rate * decay_function
            neighbourhood_radius = map_radius * decay_function
            indexes = self.get_neighbours_indexes(bmu, neighbourhood_radius)

            for index_pair in indexes:
                squared_distance = bmu.count_squared_distance(self.computional_layer[index_pair])
                influence = exp(-squared_distance / (2 * neighbourhood_radius ** 2))
                weight_difference = self.computional_layer[index_pair].count_weight_difference(random_input)
                adjustement = influence * learning_rate * weight_difference
                self.computional_layer[index_pair] = adjustement

    def predict(self, testing_set):
        predicted_values = list()

        for testing_input in testing_set:
            bmu = self.get_best_matching_unit(testing_input)
            predicted_values.append(bmu.weights[0])
        return predicted_values

    def get_best_matching_unit(self, input_unit):
        distances = list()

        for node in self.computional_layer:
            distance = node.count_euclidean_distance(input_unit)
            distances.append((node, distance))
        distances.sort(key=lambda tup: tup[1])
        return distances[0][0]

    def get_neighbours_indexes(self, bmu, radius):
        neighbours_position = list()
        neighbours_position.append((bmu.x_position, bmu.y_position))

        for node in self.computional_layer:
            squared_distance = bmu.count_squared_distance(node)
            if squared_distance <= radius ** 2:
                neighbours_position.append((node.x_position, node.y_position))
        return neighbours_position


class NodesLattice(object):

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.nodes = list()

        for row in range(rows):

            for col in range(columns):
                self.nodes.append(Node(col, row))

    def __getitem__(self, position):
        x, y = position

        for node in self.nodes:
            if node.x_position == x and node.y_position == y:
                return node

    def __setitem__(self, position, value):
        x, y = position

        for i in range(len(self.nodes)):
            if self.nodes[i].x_position == x and self.nodes[i].y_position == y:
                self.nodes[i].weights[0] += value

    def __iter__(self):
        self.actual_index = 0
        self.last_index = self.rows * self.columns - 1
        return self

    def __next__(self):
        if self.actual_index > self.last_index:
            raise StopIteration

        current_node = self.nodes[self.actual_index]
        self.actual_index += 1
        return current_node

    def count_radius(self):
        return max(self.rows, self.columns) // 2  # sigma zero


class Node(object):

    dimension = 1

    def __init__(self, column_number, row_number):
        self.x_position = column_number
        self.y_position = row_number
        self.weights = list()

        for idx in range(Node.dimension):
            self.weights.append(random())

    def count_euclidean_distance(self, input_unit):
        """This method should be written in generic way.
        For now this is implemented only for one dimensional node object"""

        return abs(self.weights[0] - input_unit['sugar'])

    def count_squared_distance(self, node):
        return (self.x_position - node.x_position) ** 2 + (self.y_position - node.y_position) ** 2

    def count_weight_difference(self, input_unit):
        return input_unit['sugar'] - self.weights[0]


class DataManager(object):

    def __init__(self):
        self.file_path = '{}/CGM.csv'.format(os.path.abspath(os.path.dirname(__file__)))

    def load_data(self):
        raw_data = self.load_csv(self.file_path)
        raw_data = raw_data[1:]
        return self.convert_dataset(raw_data)

    @staticmethod
    def load_csv(file_name):
        with open(file_name, 'r') as csv_file:
            csv_reader = reader(csv_file)
            dataset = [row for row in csv_reader if row is not None]
        return dataset

    @staticmethod
    def convert_dataset(dataset):
        converted_dataset = []

        for row in dataset:
            sample = dict()
            sample['date'] = datetime.datetime.strptime(row[1].strip(), '%Y-%m-%d %H:%M:%S')
            sample['sugar'] = int(row[2].strip())

            converted_dataset.append(sample)
        return converted_dataset

    @staticmethod
    def cross_validate_data(dataset, folds_number):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / folds_number)

        for i in range(folds_number):
            fold = list()

            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(sorted(fold, key=itemgetter('date')))
        return dataset_split


class ChartPlotter(object):

    def __init__(self):
        self.plots_dir = '{}/plots'.format(os.path.abspath(os.path.dirname(__file__)))
        self.clean_plots_directory()

    def clean_plots_directory(self):
        file_list = [file for file in os.listdir(self.plots_dir) if file.endswith('.html')]

        for file in file_list:
            os.remove(os.path.join(self.plots_dir, file))

    def plot_chart(self, data_fold, actual_values, predicted_values, plot_number):
        dates = [sample['date'].strftime('%m/%d/%Y %H:%M:%S') for sample in data_fold]

        continuous_trace = go.Scatter(
            x=dates,
            y=actual_values,
            name='Actual values',
            line=dict(
                color='rgb(205, 12, 24)',
                width=4)
        )
        dotted_trace = go.Scatter(
            x=dates,
            y=predicted_values,
            name='Predicted values',
            line=dict(
                color='rgb(22, 96, 167)',
                width=4,
                dash='dot')
        )
        traces = [continuous_trace, dotted_trace]

        layout = dict(title='Prediction of blood sugar level\n(Fold {})'.format(plot_number),
                      xaxis=dict(title='Date'),
                      yaxis=dict(title='Blood sugar level [mg/dL]'),
                      )

        fig = dict(data=traces, layout=layout)
        plot(fig, filename='{0}/plot{1}.html'.format(self.plots_dir, plot_number), auto_open=False)


class StatisticProvider(object):

    def __init__(self):
        pass

    @staticmethod
    def count_mean_absolute_percentage_error(actual_values, predicted_values):
        actual_values, predicted_values = np.array(actual_values), np.array(predicted_values)
        return np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    @staticmethod
    def print_differences_between_values(actual, predicted, fold_number):
        print('Fold {}:'.format(fold_number))
        print('ACTUAL -> PREDICTED')
        for i in range(len(predicted)):
            print('{:<3} -> {:.2f}'.format(actual[i], predicted[i]))
        print('\n\n')

    @staticmethod
    def print_statistics(scores):
        print('\nMean absolute percentage error:')

        for i in range(len(scores)):
            print('  Fold {:<2}: {:.2f}%'.format(i + 1, scores[i]))
        print('Mean accuracy: {:.2f}%'.format(100 - sum(scores) / float(len(scores))))


def create_option_parser():
    parser = ArgumentParser()

    parser.add_argument('-R', '--rows',
                        type=int,
                        required=True,
                        help='number of rows in node\'s lattice (competitive layer)')
    parser.add_argument('-C', '--columns',
                        type=int,
                        required=True,
                        help='number of columns in node\'s lattice (competitive layer)')
    parser.add_argument('-E', '--epochs',
                        type=int,
                        required=True,
                        help='Number of training\'s iterations')
    parser.add_argument('-L', '--learning_rate',
                        type=float,
                        default=0.3,
                        help='value of learning rate parameter used in network training')
    parser.add_argument('-F', '--folds',
                        type=int,
                        required=True,
                        help='number of folds used in cross validation method')
    parser.add_argument('-D', '--data_limit',
                        type=int,
                        default=2000,
                        help='limit of samples used for the algorithm')
    return parser


def main():
    seed()

    parser = create_option_parser()
    arguments = parser.parse_args()

    rows = arguments.rows
    columns = arguments.columns
    data_limit = arguments.data_limit
    folds_number = arguments.folds
    epochs = arguments.epochs
    initial_learning_rate = arguments.learning_rate

    manager = DataManager()
    plotter = ChartPlotter()
    statistic_provider = StatisticProvider()
    network = KohonenNetwork(rows, columns)  # Consider if network should be constructed inside a loop

    dataset = manager.load_data()
    dataset = dataset[:data_limit]

    folds = manager.cross_validate_data(dataset, folds_number)
    scores = list()
    counter = 1

    for fold in folds:
        training_set = list(folds)
        training_set.remove(fold)
        training_set = sum(training_set, [])
        testing_set = fold

        network.train(training_set, epochs, initial_learning_rate)

        actual_values = [record['sugar'] for record in fold]
        predicted_values = network.predict(testing_set)
        statistic_provider.print_differences_between_values(actual_values, predicted_values, counter)

        plotter.plot_chart(testing_set, actual_values, predicted_values, counter)
        counter += 1

        accuracy = statistic_provider.count_mean_absolute_percentage_error(actual_values, predicted_values)
        scores.append(accuracy)

    statistic_provider.print_statistics(scores)


if __name__ == '__main__':
    main()
