from faulthandler import disable
import sys
import os
import time
import datetime
import imutils
import numpy as np
import cv2

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from mainUi import Ui_MainWindow
from centroidtracker import CentroidTracker

#Fix pyqt with opencv error on linux
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


#Main application window
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        #Load Ui file
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #Video stream function
        self.updateComponents()


    #Set image to image label
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.ui.imageLabel.setPixmap(QPixmap.fromImage(image))


    #Set text to labels
    @pyqtSlot(str)
    def setPersonLabels(self,person):
        self.ui.current_count.setText(f"Currently Detected: {person}")
        if person=="Disabled":
            return
        if person != self.current_count_temp:
            self.current_count_temp = int(self.current_count_temp)
            person = int(person)
            if self.current_count_temp < person:
                self.current_counted_total += (person - self.current_count_temp)

        self.current_count_temp = person
        self.ui.total_count.setText(f"Total Detected: {self.current_counted_total}")


    def state_changed(self):
        if self.ui.toggleCount.isChecked():
            self.wt.disableCount = True
        else:
            self.wt.disableCount = False

        if self.ui.toggleDetection.isChecked():
            self.wt.disableDetection = True
        else:
            self.wt.disableDetection = False

    #Run thread to update images and labels
    def updateComponents(self):
        self.wt = WorkerThread(self)
        self.wt.ImageUpdate.connect(self.setImage)
        self.wt.LabelUpdate.connect(self.setPersonLabels)
        self.ui.toggleCount.stateChanged.connect(self.state_changed)
        self.ui.toggleDetection.stateChanged.connect(self.state_changed)
        self.ui.resetCount.clicked.connect(self.clicked)
        self.wt.disableCount = False
        self.wt.disableDetection = False

        self.current_count_temp = 0
        #Check if file exists then create file
        if os.path.exists("data.txt"):
            file = open("data.txt","r")
            self.total_person = int(file.read())
            file.close()
            self.ui.total_count.setText("Total Counted:" + str(self.total_person))
        else:
            self.total_person = 0
        self.current_counted_total = self.total_person

        self.wt.start()
        self.show()

    def clicked(self):
        file = open("data.txt", "w")
        file.write(str(self.current_counted_total))
        file.close()
        self.current_counted_total = 0
        self.ui.total_count.setText("Total Counted: 0")

    def closeEvent(self, event):

        file = open("data.txt", "w")
        file.write(str(self.current_counted_total))
        file.close()

    def addImageToWidget(self,cap):
        #while True:
            #QApplication.processEvents() 
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)

            #pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.imageLabel.setPixmap(pixmap)
            
            #self.ui.imageLabel.resize(pixmap.width(),pixmap.height())


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
#detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=10, maxDistance=90)



#Thread for displaying images and labels
class WorkerThread(QThread):
    ImageUpdate = pyqtSignal(QImage)
    LabelUpdate = pyqtSignal(str)
    cap = cv2.VideoCapture(0)

    def run(self):

        total_frames = 0
        lpc_count = 0
        opc_count = 0
        object_id_list = []
        while True:
            ret, frame = self.cap.read()
            frame = imutils.resize(frame, width=600)
            total_frames = total_frames + 1

            (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

            detector.setInput(blob)
            person_detections = detector.forward()
            rects = []
            for i in np.arange(0, person_detections.shape[2]):
                confidence = person_detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(person_detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    print(f"person box var {person_box}")
                    (startX, startY, endX, endY) = person_box.astype("int")
                    rects.append(person_box)
            

            boundingboxes = np.array(rects)
            boundingboxes = boundingboxes.astype(int)

            rects = non_max_suppression_fast(boundingboxes, 0.3)

            objects = tracker.update(rects)
            for (objectId, bbox) in objects.items():
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = "Person: {}".format(objectId+1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                if objectId not in object_id_list:
                    object_id_list.append(objectId)

            lpc_count = len(objects)
            opc_count = len(object_id_list)

            lpc_txt = "LPC: {}".format(lpc_count)
            opc_txt = "OPC: {}".format(opc_count)

            cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            # pixmap = QPixmap.fromImage(img)

            pixmap = img.scaled(900, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ImageUpdate.emit(pixmap)
            if self.disableCount:
                self.LabelUpdate.emit("Disabled")
            else:
                self.LabelUpdate.emit(str(lpc_count))

    def stop(self):
        self.quit()

def main():
    app = QtWidgets.QApplication(sys.argv)

    splash = QSplashScreen()
    splash.setPixmap(QPixmap('./loader.gif').scaled(100, 100))
    #splash.showMessage('<h1 style="color:white;">Welcome to use this PyQt5-SplashScreen</h1>',
     #               Qt.AlignTop | Qt.AlignHCenter, Qt.white) 
    splash.show()

    time.sleep(5)
    application = ApplicationWindow()
    #application.setFixedSize(500,500)
    application.move(100,20)
    splash.finish(application)
    #application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
