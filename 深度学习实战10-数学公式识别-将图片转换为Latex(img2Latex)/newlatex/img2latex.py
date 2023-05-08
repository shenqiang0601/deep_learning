from pix2text import Pix2Text

img_fp = 'gongshi6.png'
p2t = Pix2Text(analyzer_config=dict(model_name='mfd'))
outs = p2t(img_fp, resized_shape=600)  # 也可以使用 `p2t.recognize(img_fp)` 获得相同的结果

# 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
only_text = '\n'.join([out['text'] for out in outs])

print(only_text)