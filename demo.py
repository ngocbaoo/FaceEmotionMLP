import os
import cv2
import pickle
import numpy as np
from gui import BaseLayout
from detectors import FaceDetector
from classifiers import MultiLayerPerceptron
from datasets import homebrew
from os import path
import wx

class FaceLayout(BaseLayout):
    def __init__(self, *args, **kwargs):
        self.data_file = r'datasets\faces_training.pkl'
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Không thể mở webcam!")
            raise RuntimeError("Can not open webcam")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.emotion_list = ['neutral', 'happy', 'sad', 'surprise', 'angry', 'disgusted']
        super().__init__(*args, **kwargs)

    def _init_base_layout(self):
        self.timer = wx.Timer(self)
        self.timer.Start(1000 // 30)
        self.Bind(wx.EVT_TIMER, self._on_next_frame)
        self._init_custom_layout()

    def _create_base_layout(self):
        self.pnl = wx.Panel(self, -1, size=(self.imgWidth, self.imgHeight))
        self.pnl.SetBackgroundColour(wx.BLACK)
        self.pnl.Bind(wx.EVT_PAINT, self._on_paint)
        self.panels_vertical = wx.BoxSizer(wx.VERTICAL)
        self.panels_vertical.Add(self.pnl, 1, flag=wx.EXPAND | wx.TOP, border=1)
        self._create_custom_layout()
        self.SetMinSize((self.imgWidth, self.imgHeight))
        self.SetSizer(self.panels_vertical)
        self.Centre()

    def _create_custom_layout(self):
        self.testing = wx.RadioButton(self, -1, label="Test", style=wx.RB_GROUP)
        self.status = wx.StaticText(self, -1, "Ready")
        self.panels_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        self.panels_horizontal.Add(self.testing, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.panels_horizontal.Add(self.status, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.panels_vertical.Add(self.panels_horizontal, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.Bind(wx.EVT_RADIOBUTTON, self._on_select_test, self.testing)

    def init_algorithm(self, load_preprocessed_data=r'datasets\faces_preprocessed.pkl',
                       load_mlp=r'params\mlp.xml'):
        self.faces = FaceDetector(face_casc=r'params\haarcascade_frontalface_default.xml',
                                  left_eye_casc=r'params\haarcascade_lefteye_2splits.xml',
                                  right_eye_casc=r'params\haarcascade_righteye_2splits.xml')
        if path.isfile(load_preprocessed_data):
            (_, y_train), (_, y_test), V, m = homebrew.load_from_file(load_preprocessed_data)
            self.pca_V = V
            self.pca_m = m
            self.all_labels = np.unique(np.hstack((y_train, y_test)))
            if path.isfile(load_mlp):
                layer_sizes = np.array([self.pca_V.shape[1], 200, 100, len(self.all_labels)]) 
                self.MLP = MultiLayerPerceptron(layer_sizes, self.all_labels)
                self.MLP.load(load_mlp)
                if self.MLP.model.isTrained():
                    print("MLP loaded successfully from mlp.xml for real-time testing")
                else:
                    print("MLP failed to load from mlp.xml")
                    self.testing.Disable()
            else:
                print("Warning: No MLP file found at", load_mlp)
                self.testing.Disable()
        else:
            print("Warning: No preprocessed data found at", load_preprocessed_data)
            self.testing.Disable()

    def _acquire_frame(self):
        ret, frame = self.capture.read()
        return ret, frame

    def _process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotion_label = "Unknown"
        
        if self.testing.GetValue():
            success, processed = self.faces.process_frame(frame_rgb)
            if success:
                X_pca = np.dot(processed - self.pca_m, self.pca_V.T).astype(np.float32)
                X_pca = X_pca.reshape(1, -1)
                label = self.MLP.predict(X_pca)[0]
                emotion_label = self.emotion_list[label]
                self._update_status(emotion_label)
                
                faces = self.faces.face_casc.detectMultiScale(
                    cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), (0, 0), fx=1.0 / self.faces.scale_factor, fy=1.0 / self.faces.scale_factor),
                    scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_FIND_BIGGEST_OBJECT) * self.faces.scale_factor
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (100, 255, 0), 2)
                    cv2.putText(frame_rgb, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame_rgb

    def _on_select_test(self, event):
        print("Testing mode selected")

    def _on_exit(self, evt):
        self.capture.release()
        self.Destroy()

if __name__ == '__main__':
    app = wx.App()
    frame = FaceLayout(None, title="Face Expression Demo", fps=30,
                       imgWidth=320, imgHeight=240)
    frame.Show()
    app.MainLoop()