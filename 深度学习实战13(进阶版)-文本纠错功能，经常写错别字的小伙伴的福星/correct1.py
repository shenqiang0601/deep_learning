import sys

sys.path.append("..")

import pycorrector

if __name__ == '__main__':

    error_sentences = [
        '他是有明的侦探',
        '这场比赛我甘败下风',
        '这家伙还蛮格尽职守的',
        '报应接中迩来',
        '今天我很高形',
        '少先队员因该为老人让坐',
        '老是在较书。'
    ]
    for line in error_sentences:
        correct_sent, err = pycorrector.correct(line)
        print("{} => {} {}".format(line, correct_sent, err))