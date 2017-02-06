"""
Configuration of preprocessing, SDC model and training process, e.g.,

- `train_data` to specify multiple training folders
- `model_prefix` to specify where the trained model to be saved
- turn on one model implementation by comment on its configuration part and comment off others - yeah, not ugly, but convinient for exploration
"""

from sdc import process


## common settings
batch_size = 256
model_prefix = "models/model"
xycols = ["CenterImage", "SteeringAngle"]
xycols_left = ["LeftImage", "SteeringAngle"]
xycols_right = ["RightImage", "SteeringAngle"]
train_data = [
    ("data/t1udacity/driving_log.csv", "data/t1udacity/IMG")
]


# # nvidia model
# # 1. the default normalization used by conv2d requires input to be in range (0, 1)
# model_name="nvidia"
# image_size = (80, 160, 3)
# processors = {"CenterImage": process.yuv_normalizer(image_size)}


# vgg 16 setting
# 1. looks like a square size e.g. (80, 80) is consistently better than (80, 160)
model_name = "vgg16_pretrained"
image_size = (80, 80, 3)
processors = {"CenterImage": process.vgg_processor(image_size)}

# # vgg 16 multilayer setting
# # 1. use 80 x 80
# model_name = "vgg16_multi_layers"
# image_size = (80, 80, 3)
# processors = {"CenterImage": process.vgg_processor(image_size)}


# # comma ai setting
# model_name = "comma_ai"
# image_size = (160, 320, 3)
# processors = {"CenterImage": process.rgb_processor(image_size)}
