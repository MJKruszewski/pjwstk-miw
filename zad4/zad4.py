from __future__ import division
import random
import itertools


class RandomPredictor:
    @staticmethod
    def predict():
        return random.choice(['R', 'P', 'S'])


class MarkovChain:
    def __init__(self, order, decay=1.0):
        self.decay = decay
        self.matrix = self.create_matrix(order)

    @staticmethod
    def create_matrix(order):

        def create_keys(order):

            keys = ['R', 'P', 'S']

            for i in range((order * 2 - 1)):
                key_len = len(keys)
                for i in itertools.product(keys, ''.join(keys)):
                    keys.append(''.join(i))
                keys = keys[key_len:]

            return keys

        keys = create_keys(order)

        matrix = {}
        for key in keys:
            matrix[key] = {'R': {'prob': 1 / 3,
                                 'n_obs': 0
                                 },
                           'P': {'prob': 1 / 3,
                                 'n_obs': 0
                                 },
                           'S': {'prob': 1 / 3,
                                 'n_obs': 0
                                 }
                           }

        return matrix

    def update_matrix(self, pair, input):

        for i in self.matrix[pair]:
            self.matrix[pair][i]['n_obs'] = self.decay * self.matrix[pair][i]['n_obs']

        self.matrix[pair][input]['n_obs'] = self.matrix[pair][input]['n_obs'] + 1

        n_total = 0
        for i in self.matrix[pair]:
            n_total += self.matrix[pair][i]['n_obs']

        for i in self.matrix[pair]:
            self.matrix[pair][i]['prob'] = self.matrix[pair][i]['n_obs'] / n_total

    def predict(self, pair):
        probs = self.matrix[pair]

        if max(probs.values()) == min(probs.values()):
            return random.choice(['R', 'P', 'S'])
        else:
            return max([(i[1], i[0]) for i in probs.items()])[1]


model = RandomPredictor()
output = model.predict()
markov_model = MarkovChain(1, 0.9)
beat = {'R': 'P', 'P': 'S', 'S': 'R'}

# the first round
if input == '':

    random_predictor = RandomPredictor()
    markov_model = MarkovChain(1, 0.9)

    pair_diff2 = ''
    pair_diff1 = ''
# further rounds
# else:


    pair_diff2 = pair_diff1
    pair_diff1 = output + input


if pair_diff2 != '':
    markov_model.update_matrix(pair_diff2, input)
    output = beat[markov_model.predict(pair_diff1)]
else:
    output = random_predictor.predict()

# markov_model.matrix
# {'PP': {'P': {'n_obs': 0, 'prob': 0.3333333333333333},
#         'R': {'n_obs': 0, 'prob': 0.3333333333333333},
#         'S': {'n_obs': 0, 'prob': 0.3333333333333333}},
#  'PR': {'P': {'n_obs': 0, 'prob': 0.3333333333333333},
#         'R': {'n_obs': 0, 'prob': 0.3333333333333333},
#         'S': {'n_obs': 0, 'prob': 0.3333333333333333}},
#  'PS': {'P': {'n_obs': 0, 'prob': 0.3333333333333333},
#         'R': {'n_obs': 0, 'prob': 0.3333333333333333},
#         'S': {'n_obs': 0, 'prob': 0.3333333333333333}},
#  'RP': {'P': {'n_obs': 0, 'prob': 0.3333333333333333},
#         'R': {'n_obs': 0, 'prob': 0.3333333333333333},
#         'S': {'n_obs': 0, 'prob': 0.3333333333333333}}