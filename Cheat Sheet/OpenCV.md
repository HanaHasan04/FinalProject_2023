# OpenCV

Importing OpenCV library: ```import cv2```  

```cv2.waitKey``` pauses the execution of the script until we press a key on our keyboard. Using a parameter of 0 indicates that any keypress will un-pause the execution.  

## LOADING, DISPLAYING AND SAVING
  
```img = cv2.imread(Path)```: returns a NumPy array representing the image, (rows (height) x columns (width) x color (3)), (0,0) is top-left corner of the image  
```cv2.imshow("Title of window", img)```: displaying the actual image on our screen  
```cv2.imwrite("newimage.jpg", img)```: save image  
* Each pixel img[i,j] is a (b,g,r) array (notice that it's RGB reversed) s.t. b,g,r $\in$ [0,255]. 
*  In a grayscale image, each pixel has a value between 0 and 255, where zero corresponds to “black” and 255 corresponds to “white”. The values in between 0 and 255 are varying shades of gray, where values closer to 0 are darker and values closer to 255 are lighter.
 

```
# resize image
img = cv2.resize(img, (500,500)) 
# convert an image from one color space to another
im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
```
  

**Save image to a specific folder:**    
```
import cv2
import os
img = cv2.imread('1.jpg', 1)
path = 'D:/OpenCV/Scripts/Images'
cv2.imwrite(os.path.join(path , 'waka.jpg'), img)
cv2.waitKey(0)
```  

**Drawing on image:**    
```
# draw line given start and end points
green = (0, 255, 0)
cv2.line(canvas, (0, 0), (300, 300), green)
* last paramater is the thickness
red = (0, 0, 255)
cv2.line(canvas, (300, 0), (0, 300), red, 3)
# draw rectangle 
cv2.rectangle(canvas, (50, 200), (200, 225), red, 5)  
## negative thickness means filled shape

# draw circle given center
cv2.circle(canvas, (centerX, centerY), r, white)
```

**IMPORTANT!! EXTRACTING AND SAVING VIDEO FRAMES**  
```
import cv2
# Opens the Video file
cap= cv2.VideoCapture('C:/New/Videos/Play.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()
```  
