import numpy as np
import random
import itertools
import pandas


class MarkoPolo:
    def __init__(self):
        self.lastEvent = None

        n_ev = len(POSSIBLE_EVENTS)
        self.transition_matrix = np.zeros((n_ev, n_ev), dtype='int64')

    @staticmethod
    def update_tranistion_matrix(seq, degree=0, win=0):
        """Liczymy prawdopodobieństwo warunkowe dopasowania do łańcucha markova
        """
        n = len(POSSIBLE_EVENTS)
        out = np.zeros([n] * (degree + 1), dtype='int32')

        if isinstance(seq, str):
            seq = [(e[0], e[1]) for e in seq.split(' ')]

        for event_and_history in zip(*(seq[i:] for i in range(degree + 1))):
            eahi = [POSSIBLE_EVENTS.index(e) for e in event_and_history]

            o = out
            for i in eahi[:-1]:
                o = o[i]

            # o[eahi[-1]] += win
            o[eahi[-1]] += 1

        return out

    @staticmethod
    def probability_marginal_pair(transition_matrix):
        assert len(transition_matrix.shape) == 1
        assert transition_matrix.shape == (len(POSSIBLE_EVENTS),)

        out = np.empty((len(CHOICES),))
        out[:] = 1

        for p, (y, m) in zip(transition_matrix, POSSIBLE_EVENTS):
            yi = CHOICES.index(y)
            out[yi] += p

        return out / np.sum(out)

    @staticmethod
    def best_strategy(marginal_probs):
        assert marginal_probs.shape == (len(CHOICES),)

        outcomes = np.zeros((len(CHOICES),))

        for im, m in enumerate(CHOICES):
            '''Mnozymy sobie punktacje przez aktualna macierz tranzycji'''
            outcomes[im] = sum((
                py * SCORES[(y, m)]
                for py, y in zip(marginal_probs, CHOICES)
            ))

        i_best = np.argmax(outcomes)

        return CHOICES[i_best], outcomes[i_best]

    def get_transition_matrix(self):
        return self.transition_matrix

    def predict(self):
        last_event = self.lastEvent
        counts = MarkoPolo.get_transition_matrix(self)

        '''Jak juz jakas historie mamy to z niej korzystamy w innym przypadku na start bierzemy losowy wybor'''
        if last_event is not None:
            lei = POSSIBLE_EVENTS.index(last_event)
            p_1 = MarkoPolo.probability_marginal_pair(counts[lei, :])

            best_strategy, expected_outcome = MarkoPolo.best_strategy(p_1)
        else:
            p_1 = np.empty((len(CHOICES),))
            p_1[:] = 1. / 3
            best_strategy, expected_outcome = random.choice(CHOICES), 0

        predicted_plays = dict(zip(CHOICES, p_1))

        return best_strategy, expected_outcome, predicted_plays

    def log_event(self, player_choice, bot_choice, win=0):
        current_event = (player_choice, bot_choice)
        last_event = self.lastEvent

        if last_event is not None:
            counts = MarkoPolo.get_transition_matrix(self)
            counts += MarkoPolo.update_tranistion_matrix([last_event, current_event], degree=1, win=win)
            self.transition_matrix = counts

        self.lastEvent = current_event


def score(p, b):
    if p == b:
        print("REMIS")

        return 0
    elif p == 'K' and b == 'P':
        return 1
    elif p == 'K' and b == 'N':
        return -1
    elif p == 'P' and b == 'K':
        return -1
    elif p == 'P' and b == 'N':
        return 1
    elif p == 'N' and b == 'K':
        return 1
    elif p == 'N' and b == 'P':
        return -1

    return -1


CHOICES = 'KPN'
POSSIBLE_EVENTS = list(itertools.product(CHOICES, CHOICES))
SCORES = {
    ('K', 'K'): 0,
    ('K', 'P'): 1,
    ('K', 'N'): -1,

    ('P', 'K'): -1,
    ('P', 'P'): 0,
    ('P', 'N'): 1,

    ('N', 'K'): 1,
    ('N', 'P'): -1,
    ('N', 'N'): 0
}

marko = MarkoPolo()
player_pick = ''

while player_pick != 'S':
    player_pick = input("Kamień(K), papier(P), nożyce(N), stop(S): ")

    if player_pick == 'S':
        continue

    best_strategy, expected_outcome, predicted_plays = marko.predict()
    scoreResult = score(player_pick, best_strategy)
    marko.log_event(player_pick, best_strategy, scoreResult)

    print('______')
    print('BOT')

    print('best_strategy: ', best_strategy)
    print('expected_outcome: ', expected_outcome)
    print('predicted_plays: ', predicted_plays)

    print('______')
    df = pandas.DataFrame(marko.get_transition_matrix(), columns=['K', 'K', 'K', 'P', 'P', 'P', 'N', 'N', 'N'], index=['K', 'P', 'N', 'K', 'P', 'N', 'K', 'P', 'N'])
    print(df.to_string())
    print('______')

    if scoreResult == 1:
        print("Wygrywa bot")
    else:
        print("Wygrywa Gracz")

print("Wynik: ", np.sum(marko.get_transition_matrix()))