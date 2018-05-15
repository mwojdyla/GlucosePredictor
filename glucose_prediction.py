#  REMEMBER TO SET CREDENTIALS FILE IN PLOTLY FOR ONLINE PLOTTING
import datetime
import numpy as np
import os
import plotly

from argparse import ArgumentParser
from csv import reader
from math import exp, log
from operator import itemgetter
from plotly import graph_objs as go
from plotly import plotly as py
from plotly.offline import plot
from random import seed, random, randint, randrange
from tabulate import tabulate


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
        radius = max(self.rows, self.columns) / 2  # sigma zero
        return radius if radius > 1 else 1.1


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
        if folds_number < 2:
            raise Exception('Invalid number of folds occured!')

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
        self.images_dir = '{}/images'.format(os.path.abspath(os.path.dirname(__file__)))
        self.clean_plots_directory()

    def clean_plots_directory(self):
        file_list = [file for file in os.listdir(self.plots_dir) if file.endswith('.html')]

        for file in file_list:
            os.remove(os.path.join(self.plots_dir, file))

    def plot_chart(self, data_fold, actual_values, predicted_values, plot_number):
        traces = self.make_traces(data_fold, actual_values, predicted_values)
        layout = self.make_layout_for_offline_plotting(plot_number)
        fig = dict(data=traces, layout=layout)
        plot(fig, filename='{0}/plot{1}.html'.format(self.plots_dir, plot_number), auto_open=False)

    def plot_chart_online(self, data_fold, actual_values, predicted_values, plot_number):
        traces = self.make_traces(data_fold, actual_values, predicted_values)
        layout = self.make_layout_for_online_plotting(plot_number)
        fig = go.Figure(data=traces, layout=layout)
        py.image.save_as(fig, filename='{0}/plot_image{1}.png'.format(self.images_dir, plot_number))

    @staticmethod
    def make_traces(data_fold, actual_values, predicted_values):
        dates = [sample['date'].strftime('%m/%d/%Y %H:%M:%S') for sample in data_fold]

        continuous_trace = go.Scatter(
            x=dates,
            y=actual_values,
            name='Actual value',
            line=dict(
                color='rgb(205, 12, 24)',
                width=4)
        )
        dotted_trace = go.Scatter(
            x=dates,
            y=predicted_values,
            name='Predicted value',
            line=dict(
                color='rgb(22, 96, 167)',
                width=4,
                dash='dot')
        )
        return [continuous_trace, dotted_trace]

    @staticmethod
    def make_layout_for_online_plotting(plot_number):
        return go.Layout(
            title='Prediction of blood sugar level\n(Fold {})'.format(plot_number),
            xaxis=dict(title='Date'),
            yaxis=dict(title='Blood sugar level [mg/dL]'),
            autosize=False,
            width=1500,
            height=600,
            legend=dict(font=dict(size=14))
        )

    @staticmethod
    def make_layout_for_offline_plotting(plot_number):
        return dict(
            title='Prediction of blood sugar level\n(Fold {})'.format(plot_number),
            xaxis=dict(title='Date'),
            yaxis=dict(title='Blood sugar level [mg/dL]'),
            legend=dict(font=dict(size=14))
        )


class StatisticProvider(object):

    def __init__(self):
        pass

    @staticmethod
    def count_mean_absolute_percentage_error(actual_values, predicted_values):
        actual_values, predicted_values = np.array(actual_values), np.array(predicted_values)
        return np.mean(np.abs((actual_values - predicted_values) / actual_values) * 100)

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

        print('\nPrediction accuracy:')

        for i in range(len(scores)):
            print('  Fold {:<2}: {:.2f}%'.format(i + 1, 100 - scores[i]))

        print('\nMean accuracy: {:.2f}%'.format(100 - sum(scores) / float(len(scores))))


class TableGenerator(object):

    def __init__(self):
        self.latex_tables_dir = '{}/latex_tables'.format(os.path.abspath(os.path.dirname(__file__)))
        self.latex_table_template = \
            '''
            \\begin{table}[]
            \centering
            tabular_content
            \caption{My caption}
            \end{table}
            '''
        self.clean_latex_tables_directory()

    def clean_latex_tables_directory(self):
        file_list = [file for file in os.listdir(self.latex_tables_dir) if file.endswith('.txt')]

        for file in file_list:
            os.remove(os.path.join(self.latex_tables_dir, file))

    def make_latex_table(self, actual_values, predicted_values, fold_number):
        content = self.generate_tabular_content(actual_values, predicted_values)
        self.latex_table_template = self.latex_table_template.replace('tabular_content', content)
        self.write_table_to_file(fold_number)

    def write_table_to_file(self, fold_number):
        file_name = '{0}/table{1}.txt'.format(self.latex_tables_dir, fold_number)
        with open(file_name, 'w') as txt_file:
            txt_file.write(self.latex_table_template)

    @staticmethod
    def generate_tabular_content(actual_values, predicted_values):
        actual_val_key = 'Actual values [mg/dL]'
        predicted_val_key = 'Predicted values [mg/dL]'
        sample_num_key = 'Sample number'
        sample_numbers = list(range(1, len(actual_values) + 1))

        latex_tabular = tabulate(
            {
                sample_num_key: sample_numbers,
                actual_val_key: actual_values,
                predicted_val_key: predicted_values
            },
            headers='keys',
            tablefmt='latex',
            numalign='center'
        )
        return TableGenerator.modify_table_content(latex_tabular)

    @staticmethod
    def modify_table_content(latex_table):
        modified_table = latex_table.replace('ccc', '|c|c|c|')
        modified_table = modified_table.replace('Actual values [mg/dL]', '\\textbf{Actual values [mg/dL]}')
        modified_table = modified_table.replace('Predicted values [mg/dL]', '\\textbf{Predicted values [mg/dL]}')
        modified_table = modified_table.replace('Sample number', '\\textbf{Sample number}')
        return modified_table


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
    parser.add_argument('--online_plotting',
                        action='store_true',
                        help='enable using plotly in online mode')
    return parser


def main():
    seed()

    parser = create_option_parser()
    arguments = parser.parse_args()

    rows = arguments.rows
    columns = arguments.columns
    epochs = arguments.epochs
    initial_learning_rate = arguments.learning_rate
    folds_number = arguments.folds
    data_limit = arguments.data_limit
    enable_online_plotting = arguments.online_plotting

    manager = DataManager()
    plotter = ChartPlotter()
    generator = TableGenerator()
    statistic_provider = StatisticProvider()

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

        network = KohonenNetwork(rows, columns)
        network.train(training_set, epochs, initial_learning_rate)

        actual_values = [record['sugar'] for record in testing_set]
        predicted_values = network.predict(testing_set)
        statistic_provider.print_differences_between_values(actual_values, predicted_values, counter)

        if enable_online_plotting:
            plotter.plot_chart_online(testing_set, actual_values, predicted_values, counter)
        else:
            plotter.plot_chart(testing_set, actual_values, predicted_values, counter)

        generator.make_latex_table(actual_values, predicted_values, counter)

        accuracy = statistic_provider.count_mean_absolute_percentage_error(actual_values, predicted_values)
        scores.append(accuracy)
        counter += 1

    statistic_provider.print_statistics(scores)


if __name__ == '__main__':
    main()
