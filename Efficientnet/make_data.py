import numpy as np
import cv2
import time
import os
0
# Label: 00000 là ko cầm tiền, còn lại là các mệnh giá
label = "20000"

cap = cv2.VideoCapture(0)

#Count from 60th frame
i = 0
while True:
    # Capture frame-by-frame
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)

    
    cv2.imshow('frame', frame)

    # Save data from the frame of 60 to 932
    if i >= 60 and i < 1000:
        print("Số ảnh capture =", i - 60)
        #Create folder
        data_dir = 'data/' + str(label)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        cv2.imwrite(os.path.join(data_dir, str(i) + ".png"), frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
