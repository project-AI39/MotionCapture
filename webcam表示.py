import cv2

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        camera_parameter = ['CAP_PROP_FRAME_WIDTH',
        'CAP_PROP_FRAME_HEIGHT',
        'CAP_PROP_FOURCC',
        'CAP_PROP_BRIGHTNESS',
        'CAP_PROP_CONTRAST',
        'CAP_PROP_SATURATION',
        'CAP_PROP_HUE',
        'CAP_PROP_GAIN',
        'CAP_PROP_EXPOSURE',]

        for x in range(9):
            print(camera_parameter[x], '=', capture.get(x))
            
        break

capture.release()
cv2.destroyAllWindows()