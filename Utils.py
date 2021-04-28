import os
import imageio

png_dir = './output_ims/not-needed/experiment15-finished/'
images = []
paths = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png') and f'{25}-{0}-' in file_name:
        file_path = os.path.join(png_dir, file_name)
        paths.append(file_path)
paths = sorted(paths)
for file_path in paths:
    images.append(imageio.imread(file_path))
imageio.mimsave(png_dir+f'{25}-{0}-gif.gif', images)