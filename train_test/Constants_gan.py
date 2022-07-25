'''
Editor 2021 10 18
'''

train_interval = 1
ratio_gan2seg = 100
mse_ratio = 1.0
gpu_index = '0'
discriminator_type = 'image'
batch_size = 4
dataset = 'NUCLEI'
is_test = False
learning_rate = 1e-4
beta1 = 0.5
iters = 30000
early_stop = 1000
visual_samples = './log/visual_samples/'
