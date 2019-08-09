#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/8/9 20:38
"""

import os
import pygame

DATA_DIR = "../../data/chinese-char/img"
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

fonts_type = [
    "../../data/chinese-char/ttfs/msyh.ttf",
    "../../data/chinese-char/ttfs/nings.ttf",
    "../../data/chinese-char/ttfs/skai.ttf",
    "../../data/chinese-char/ttfs/wens.ttf",
    "../../data/chinese-char/ttfs/yaso.ttf",
]

pygame.init()
font = pygame.font.Font(fonts_type[4], 128)


def load_chinese_chars():
    filepath = "../../data/chinese-char/chinese.txt"
    symbols = []
    with open(filepath, encoding="utf8") as fr:
        for line in fr:
            line = line.strip()
            if len(line) > 0:
                symbols.extend([x for x in line])

    return symbols


def trans_char_to_img():
    symbols = load_chinese_chars()
    for i, word in enumerate(symbols):
        rend = font.render(word, True, (0, 0, 0), (255, 255, 255))
        save_path = DATA_DIR + "/yaso-%d.jpg" % i
        pygame.image.save(rend, save_path)


if __name__ == '__main__':
    trans_char_to_img()
