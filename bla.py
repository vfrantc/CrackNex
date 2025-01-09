import numpy as np
from PIL import Image


if __name__ == '__main__':
    classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    mask = Image.fromarray(np.array(Image.open('/home/vladimirfrants9/mCrackNex/Datasets_CrackNex/LCSD/SegmentationClass/LCSD_0095.png')))
    bla = np.array(mask)
    print(bla[175:200, 125:150])
    print(mask)
    classes = set(np.unique(mask)) & classes
    print('unique thing: ', np.unique(mask))
    print(classes)

    classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    mask = Image.fromarray(np.array(Image.open('/home/vladimirfrants9/mCrackNex/Datasets_CrackNex/LCSD/SegmentationClass/LCSD_0227.png')))
    bla = np.array(mask)
    print(bla[175:200, 125:150])
    print(mask)
    classes = set(np.unique(mask)) & classes
    print('unique thing: ', np.unique(mask))
    print(classes)