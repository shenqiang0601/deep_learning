import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image

# 目前支持单人的漫画生成
img_cartoon = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon_compound-models',
                       device='cpu')
# 图像本地路径
img_path = 'new_image.png'

# img = Image.open(img_path) # 调整图片大小为800x640
# new_size = (480, 270)
# img = img.resize(new_size) # 保存修改后的图片
# img.save("new_image.png")

result = img_cartoon(img_path)
cv2.imwrite('result21.png', result[OutputKeys.OUTPUT_IMG])
print('完成!')