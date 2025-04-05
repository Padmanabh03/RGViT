import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_camera_image(image_path):
    img = Image.open(image_path)
    img_np = np.array(img)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_np)
    plt.axis('off')
    plt.title('Front Camera Image (CAM_FRONT)')
    plt.show()