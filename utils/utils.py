import csv
import numpy as np
from PIL import Image
def show_error_image():
    tag=["A","B","C","D","E","F","G","H"]
    with open('../dataset/validation_error.csv') as f:
        reader = csv.reader(f)
        count = 0
        for line in reader:
            count+=1

            line=list(line)
            error_image=np.array(line[:-2],dtype=float)
            if len(error_image)!=784:
                raise TypeError("Error image shape")
            error_image = 255 * error_image
            error_image=np.reshape(error_image,(28,28),order='F')
            # print error_image
            # print (error_image.shape)


            image = Image.fromarray(error_image)
            print ("prev:"+tag[int(line[-2])]+".real:"+tag[int(line[-1])])
            image.show()
            str = raw_input("go on?[y/n]")
            if str == 'n':
                break


if __name__ == '__main__':
    show_error_image()

