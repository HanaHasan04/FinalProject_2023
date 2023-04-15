import cv2

# Load the image
img = cv2.imread("C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject/Images/Disappointment/S16-T2-D1-C1-3__62.jpg")

# Get the size of the image
height, width = img.shape[:2]

# # Resize the image to a fixed size
# fixed_size = (224,224)
# resized_img = cv2.resize(img, fixed_size, interpolation = cv2.INTER_AREA)
#
# cv2.imshow("Hi", resized_img)
# cv2.waitKey(0)



# Crop the image to a fixed size
fixed_size = (1080,1080)
height, width = img.shape[:2]
start_x = int(width/2 - fixed_size[0]/2)
start_y = int(height/2 - fixed_size[1]/2)
cropped_img = img[start_y:start_y+fixed_size[1], start_x:start_x+fixed_size[0]]

# Resize the image to a fixed size
fixed_size = (224,224)
resized_img = cv2.resize(cropped_img, fixed_size, interpolation = cv2.INTER_AREA)

# Resize the image to a fixed size
fixed_size = (224,224)
resized_img1 = cv2.resize(img, fixed_size, interpolation = cv2.INTER_AREA)
cv2.imshow("Hi", resized_img1)
cv2.waitKey(0)


# cv2.imshow("Hi", img)
# cv2.waitKey(0)
cv2.imshow("Hi", resized_img)
cv2.waitKey(0)


print("The size of the image is:", width, "x", height)
