import os
import numpy
files_train="/dataset/public/ImageNetOrigin/train"
files_val="/dataset/public/ImageNetOrigin/val"
for filename in files_train:
    portion = os.path.splitext(filename)
    if portion[1] == ".JPEG":
        newname = portion[0] + ".jpg"
        os.rename(filename, newname)



for filename in files_val:
    portion = os.path.splitext(filename)
    if portion[1] == ".JPEG":
        newname = portion[0] + ".jpg"
        os.rename(filename, newname)