from skimage import io
import matplotlib.pyplot as plt
im = io.imread('./25.jpg')
plt.imshow(im)
plt.show()