# -*- coding: utf-8 -*-
""" main.py """

from config import CFG
from cnn_lstm import CNN_LSTM


def run():
    """Builds model, loads data, trains and evaluates"""
    model = CNN_LSTM(CFG)
    model.load_data()
#    model.load_test_data()
    model.build()
#    model.train()
    model.evaluate()


if __name__ == '__main__':
    print("\n--------------------------------------------------------------------\n")
    run()