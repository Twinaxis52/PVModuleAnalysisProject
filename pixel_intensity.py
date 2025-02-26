import numpy as np
import matplotlib.pyplot as plt
import cv2


# accepts lists or single images
# returns list of the gray image then average, median, std dev, variance
def pixel_intensity(input_image, path=None):
    output = None
    if type(input_image) == list:
        for image in input_image:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            average = np.average(gray)
            median = np.median(gray)
            std_dev = np.std(gray)
            variance = np.var(gray)
            plt.hist(x=image.ravel(), bins=256, range=[0,256], color='crimson')
            plt.title("Histogram of Pixel Intensity", color='crimson')
            plt.ylabel("Number of pixels", color='crimson')
            plt.xlabel("Pixel Intensity", color='crimson')
            if path:
                plt.savefig(path)
            stats = [average, median, std_dev, variance]

            if output == None:
                output = np.array([stats])
            else:
                output.append(stats)
            
    else:
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        average = np.average(gray)
        median = np.median(gray)
        std_dev = np.std(gray)
        variance = np.var(gray)
        plt.figure().set_figwidth(7)
        plt.hist(x=input_image.ravel(), bins=256, range=[0,256], color='crimson')
        plt.title("Histogram of Pixel Intensity", color='crimson')
        plt.ylabel("Number of pixels", color='crimson')
        plt.xlabel("Pixel Intensity", color='crimson')
        if path:
            plt.savefig(path)
        stats = [average, median, std_dev, variance]
        output = np.array([stats])
        
    return output

