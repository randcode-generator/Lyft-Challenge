import scipy.misc
from PIL import Image
import numpy as np

rgb_image = scipy.misc.imread("0.png")
rgb_image = scipy.misc.toimage(rgb_image)

width = 160
height = 64

startx = 0
starty = 264

a=[]
for i in range(0, 5):
    b=[]
    for j in range(0, 4):
        left = startx + (width * i)
        top = starty + (height * j)
        right = left + width
        bottom = top + height
        coords = (left, top, right, bottom)
        print(coords)
        img = rgb_image.crop(coords)
        b.append(np.array(img))
        #scipy.misc.imsave("croped_" + str(j) + ".png", img)
    a.append(b)


h = []
for i in a:
    h.append(np.vstack(i))

scipy.misc.imsave("all.png", np.hstack(h))
