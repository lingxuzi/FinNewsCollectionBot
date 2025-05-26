# -*- coding: utf-8 -*-
"""主程序入口"""
from ai.trend.models import get_trainer


def train():
    get_trainer()()

if __name__ == '__main__':
    train()