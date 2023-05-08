from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

# ModelScope Library >= 1.2.0
ocr_recognize = pipeline(Tasks.ocr_recognition, model='damo/ofa_ocr-recognition_handwriting_base_zh', model_revision='v1.0.1')
result = ocr_recognize('https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/ocr/ocr_handwriting_demo.png')
print(result[OutputKeys.TEXT])