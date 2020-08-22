import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('./watch.jpg',cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.avi', fourcc, 20.0, (640,480))

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release
cv2.destroyAllWindows()

# cv2.imshow('watch',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.plot([50,100],[80,100],'c', linewidth=5)
# plt.show()