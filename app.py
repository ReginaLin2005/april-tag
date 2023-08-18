import cv2 as cv
import numpy as np
import copy
from pupil_apriltags import Detector

# Initializing camera
device = 0
cap = cv.VideoCapture(device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

# Initializing detector
at_detector = Detector(
   families="tag16h5",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

# Label tags
def draw_tags(image,tags):
   for tag in tags:
      tag_family = tag.tag_family
      tag_id = tag.tag_id
      center = tag.center
      corners = tag.corners

      center = (int(center[0]), int(center[1]))
      corner_01 = (int(corners[0][0]), int(corners[0][1]))
      corner_02 = (int(corners[1][0]), int(corners[1][1]))
      corner_03 = (int(corners[2][0]), int(corners[2][1]))
      corner_04 = (int(corners[3][0]), int(corners[3][1]))

      # 中心
      cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

      # 各辺
      cv.line(image, (corner_01[0], corner_01[1]),
               (corner_02[0], corner_02[1]), (255, 0, 0), 2)
      cv.line(image, (corner_02[0], corner_02[1]),
               (corner_03[0], corner_03[1]), (255, 0, 0), 2)
      cv.line(image, (corner_03[0], corner_03[1]),
               (corner_04[0], corner_04[1]), (0, 255, 0), 2)
      cv.line(image, (corner_04[0], corner_04[1]),
               (corner_01[0], corner_01[1]), (0, 255, 0), 2)

      # タグファミリー、タグID
      # cv.putText(image,
      #            str(tag_family) + ':' + str(tag_id),
      #            (corner_01[0], corner_01[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
      #            0.6, (0, 255, 0), 1, cv.LINE_AA)
      cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                  cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

   return image

while True:

   # カメラキャプチャ #####################################################
   ret, image = cap.read()
   if not ret:
      break
   debug_image = copy.deepcopy(image)

   # 検出実施 #############################################################
   image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
   tags = at_detector.detect(
      image,
      estimate_tag_pose=False,
      camera_params=None,
      tag_size=None,
   )

   
   # 描画 ################################################################
   debug_image = draw_tags(debug_image, tags)

   # キー処理(ESC：終了) #################################################
   key = cv.waitKey(1)
   if key == 27:  # ESC
      break

   # 画面反映 #############################################################
   cv.imshow('AprilTag Detect Demo', debug_image)

cap.release()
cv.destroyAllWindows()

result = at_detector.detect(image)
print(result)
