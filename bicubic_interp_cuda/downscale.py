import cv2 as cv

def main():
    imagename = input("Image name: ")
    image = cv.imread(imagename)
    h, w , _ = image.shape
    image = cv.resize(image, (int(w/2), int(h/2)))
    cv.imwrite("downscaled.png", image)

if __name__ == "__main__":
    main()