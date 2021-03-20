import matplotlib.pyplot as plt
import glob
import sys
from planetaryimage import PDS3Image
dirIN = ((sys.argv)[1])
dirOUT = ((sys.argv)[2])
ext = '.IMG'
files = glob.glob(dirIN + '*' + ext)
files = [x[len(dirIN):(len(x)-len(ext))] for x in files]

print(files)
counter = 0
num = len(files)
for i in files:
    counter += 1
    print(i+' Done '+ str((float(counter/num))*100) + '%')
    image = PDS3Image.open(dirIN + i + ext)
    plt.imsave(dirOUT + i + '.png', image.image, cmap='gray')