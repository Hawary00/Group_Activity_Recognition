import os
import cv2

# First, let's verify the base path exists
base_path = r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets"
print(f"Base directory exists: {os.path.exists(base_path)}")

# List contents of the directory if it exists
if os.path.exists(base_path):
    print("\nContents of the directory:")
    for item in os.listdir(base_path):
        print(item)



import os
print(os.path.exists(r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_/videos/4/24745/24740.jpg"))



from PIL import Image

try:
    img = Image.open(r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_/videos/4/24745/24740.jpg")
    img.show()
except Exception as e:
    print(f"Error opening image: {e}")




# print(os.listdir(r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_/videos/4/24745/24740.jpg"))