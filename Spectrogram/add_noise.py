import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from scipy import signal
from scipy.io import wavfile
import cv2

# Read in wav file and display
sample_rate, samples = wavfile.read('test.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

    
img = spectrogram


"""
Method 1: Gaussian noise
"""
## source: https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a

mean = 0
var = 10
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (img.shape[0],img.shape[1])) 

noisy_image = np.zeros(img.shape, np.float32)

if len(img.shape) == 2:
    noisy_image = img + gaussian
else:
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian

cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy_image = noisy_image.astype(np.uint8)

cv2.imshow("Gaussian Noise", gaussian)
plt.imshow(noisy_image)
plt.show()

cv2.waitKey(0)






"""
Method 2: add random noise
"""

img = spectrogram
##source: https://jingyan.baidu.com/article/c910274bab6baecd361d2df2.html
# define a function
def f(i):
    if i<0:
        return 0
    elif i>255:
        return 255
    else:
        return i

c = []
for i in range(img.shape[0]):
    d=[]
    for j in range(img.shape[1]):
        hh=f(img[i,j]-random.uniform(-1,1)*100)
        d.append(hh)
    c.append(d)
c=np.array(c)

noisy = c+spectrogram

cv2.imshow("random noise",c)
plt.imshow(c)
plt.show()
plt.imshow(noisy)
plt.show()





