# -*- coding: utf-8 -*-

import sys

sys.path.append("..")
from pycorrector.macbert.macbert_corrector import MacBertCorrector


def use_origin_transformers():
    # 原生transformers库调用
    import operator
    import torch
    from transformers import BertTokenizer, BertForMaskedLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
    model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
    model.to(device)

    texts = ["今天新情很好", "你找到你最喜欢的工作，我也很高心。", "我不唉“看 琅擤琊榜”"]

    text_tokens = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**text_tokens)

    def get_errors(corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if ori_char in [' ', '“', '”', '‘', '’', '\n', '…', '—', '擤']:
                # add unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if i >= len(corrected_text):
                break
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    result = []
    for ids, (i, text) in zip(outputs.logits, enumerate(texts)):
        _text = tokenizer.decode((torch.argmax(ids, dim=-1) * text_tokens.attention_mask[i]),
                                 skip_special_tokens=True).replace(' ', '')
        corrected_text, details = get_errors(_text, text)
        print(text, ' => ', corrected_text, details)
        result.append((corrected_text, details))
    print(result)
    return result


if __name__ == '__main__':
    # 原生transformers库调用
    use_origin_transformers()

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

    m = MacBertCorrector()
    for line in error_sentences:
        correct_sent, err = m.macbert_correct(line)
        print("query:{} => {} err:{}".format(line, correct_sent, err))