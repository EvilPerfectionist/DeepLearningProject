import os

video_name = 'PussnToots'
path = '/home/leon/DeepLearning/Project/Dataset/' + video_name + '/'

for count, filename in enumerate(os.listdir(path)):
        dst ="raw_image_" + str(count) + ".png"
        src = path + filename
        dst = path + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
