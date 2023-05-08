import gradio as gr
import operator
import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")


def ai_text(text):
    with torch.no_grad():
        outputs = model(**tokenizer([text], padding=True, return_tensors='pt'))

    def to_highlight(corrected_sent, errs):
        output = [{"entity": "纠错", "word": err[1], "start": err[2], "end": err[3]} for i, err in
                  enumerate(errs)]
        return {"text": corrected_sent, "entities": output}

    def get_errors(corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
                # add unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if i >= len(corrected_text):
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    _text = tokenizer.decode(torch.argmax(outputs.logits[0], dim=-1), skip_special_tokens=True).replace(' ', '')
    corrected_text = _text[:len(text)]
    corrected_text, details = get_errors(corrected_text, text)
    print(text, ' => ', corrected_text, details)
    return to_highlight(corrected_text, details), details


if __name__ == '__main__':
    print(ai_text('少先队员因该为老人让坐'))

    examples = [
        ['真麻烦你了。希望你们好好的跳无'],
        ['少先队员因该为老人让坐'],
        ['他是有明的侦探'],
        ['今天心情很不搓'],
        ['他法语说的很好，的语也不错'],
        ['这场比赛我甘败下风'],
    ]

    gr.Interface(
        ai_text,
        inputs="textbox",
        outputs=[
            gr.outputs.HighlightedText(
                label="Output",
                show_legend=True,
            ),
            gr.outputs.JSON(
                label="JSON Output"
            )
        ],
        title="中文纠错模型",
        description="输入一段话，判断这段话中是否有错别字或语法错误",
        article="Link to <a href='https://github.com/shibing624/pycorrector' style='color:blue;' target='_blank\'>Github REPO</a>",
        examples=examples
    ).launch()