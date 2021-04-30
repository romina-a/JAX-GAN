import os
import imageio
import re

png_dir = './output_ims/experiment25-viz/5678/'
images = []
paths = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png') and f'-{"state1"}' in file_name:
        file_path = os.path.join(png_dir, file_name)
        paths.append(file_path)
paths = sorted(paths, key=lambda x: int(re.findall(r'\d+', x)[-2]))
for file_path in paths:
    images.append(imageio.imread(file_path))
imageio.mimsave(png_dir+f'{25}-{0}-state1.gif', images)
