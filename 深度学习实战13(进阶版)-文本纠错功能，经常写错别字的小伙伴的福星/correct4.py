import sys

sys.path.append("..")
from pycorrector.t5.t5_corrector import T5Corrector

if __name__ == '__main__':
    # pycorrector封装调用
    error_sentences = [
        '他是有明的侦探',
        '这场比赛我甘败下风',
        '这家伙还蛮格尽职守的',
        '报应接中迩来',
        '今天我很高形',
        '少先队员因该为老人让坐',
        '老是在较书。'
    ]

    m = T5Corrector()
    res = m.batch_t5_correct(error_sentences)
    for line, r in zip(error_sentences, res):
        correct_sent, err = r[0], r[1]
        print("query:{} => {} err:{}".format(line, correct_sent, err))