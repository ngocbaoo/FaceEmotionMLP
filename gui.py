import wx
import cv2
import numpy as np

class BaseLayout(wx.Frame):
    def __init__(self, parent, title, fps=15, imgWidth=480, imgHeight=360):
        super().__init__(parent, title=title)
        self.fps = fps
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.frame = None
        self.bmp = None
        self._init_base_layout()
        self._create_base_layout()
        self.init_algorithm()
        self.Bind(wx.EVT_CLOSE, self._on_exit)

    def _init_base_layout(self):
        pass

    def _init_custom_layout(self):
        pass

    def _create_base_layout(self):
        pass

    def _create_custom_layout(self):
        pass

    def init_algorithm(self):
        pass

    def _acquire_frame(self):
        pass

    def _process_frame(self, frame):
        return frame

    def _update_status(self, label):
        self.status.SetLabel(str(label))

    def _on_next_frame(self, event):
        ret, frame = self._acquire_frame()
        if ret:
            frame = self._process_frame(frame)
            self.frame = cv2.resize(frame, (self.imgWidth, self.imgHeight))
            self.bmp = wx.Bitmap.FromBuffer(self.imgWidth, self.imgHeight, self.frame)
            self.Refresh()

    def _on_paint(self, event):
        dc = wx.BufferedPaintDC(self.pnl)
        if self.bmp:
            dc.DrawBitmap(self.bmp, 0, 0)

    def _on_exit(self, event):
        self.Destroy()