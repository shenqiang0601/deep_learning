import torch
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

task = Tasks.text_to_image_synthesis
model_id = 'modelscope/small-stable-diffusion-v0'
# 基础调用
pipe = pipeline(task=task, model=model_id, model_revision='v1.0.2')
output = pipe({'text': 'an apple'})