import sys

import cv2.cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from utils import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from mainwindow import Ui_MainWindow
from PyQt5.QtGui import QPixmap, QImage
from filter import *
import time


class MainWindow:
    def __init__(self):
        self.original_image = None
        self.save_image = {"filter": 0, "old": 0, "b_value": 1, "v_value": None, "k_value": None, "s_value": 0}
        self.effect_image = None
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)

        self.uic.btnSelectImage.clicked.connect(self.load_image)
        self.uic.listView.currentItemChanged.connect(self.update_filter)
        self.uic.listView2.currentItemChanged.connect(self.update_old)
        self.uic.slider_brightness.valueChanged.connect(self.update_brightness_contrast)
        self.uic.slider_blur.valueChanged.connect(self.update_blur)
        self.uic.slider_vegnette.valueChanged.connect(self.update_vignette)
        self.uic.slider_saturation.valueChanged.connect(self.update_saturation)
        self.uic.btnReset.clicked.connect(self.reset_effect)
        self.uic.btnSaveImage.clicked.connect(self.save_to)

    def load_image(self):
        img_path = QFileDialog.getOpenFileName(filter="*.jpg *.png *jpeg")
        if img_path[0] == "":
            return
        self.uic.slider_brightness.setValue(50)
        self.uic.slider_blur.setValue(-1)
        self.uic.slider_vegnette.setValue(99)
        self.uic.slider_saturation.setValue(0)
        self.uic.listView.setCurrentRow(0)
        self.uic.listView2.setCurrentRow(0)
        self.original_image = cv2.imread(img_path[0])
        self.effect_image = self.original_image
        img_show = resize(self.original_image)
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        img_show = QImage(img_show, img_show.shape[1], img_show.shape[0], img_show.strides[0], QImage.Format_RGB888)
        self.uic.showImage.setPixmap(QtGui.QPixmap.fromImage(img_show))

    def update(self, image):
        if self.save_image["filter"] != 0:
            image = eval(LIST_FILTER[self.save_image["filter"]] + "(image)")
        else:
            image = self.original_image
        image = blending_image(image, self.save_image["old"])
        image = brightness_contrast(image, self.save_image["b_value"])
        image = vignette(image, self.save_image["v_value"])
        image = gaussian_blur(image, self.save_image["k_value"])
        image = change_saturation(image, self.save_image["s_value"])
        return image

    def show_image(self, image):
        tmp_img = resize(image)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        tmp_img = QImage(tmp_img, tmp_img.shape[1], tmp_img.shape[0], tmp_img.strides[0], QImage.Format_RGB888)
        self.uic.showImage.setPixmap(QtGui.QPixmap.fromImage(tmp_img))

    def update_filter(self):
        i = self.uic.listView.currentRow()
        if i == 0:
            if self.original_image is not None:
                self.save_image["filter"] = 0
                self.effect_image = self.update(self.original_image)
                self.show_image(self.effect_image)
        else:
            if self.original_image is not None:
                self.save_image["filter"] = i
                self.effect_image = self.update(self.original_image)
                self.show_image(self.effect_image)

    def update_brightness_contrast(self):
        b_value = self.uic.slider_brightness.value()
        if self.original_image is not None:
            self.save_image["b_value"] = b_value/50
            self.effect_image = self.update(self.original_image)
            self.show_image(self.effect_image)

    def update_vignette(self):
        v_value = self.uic.slider_vegnette.value()
        if self.original_image is not None:
            if v_value == 99:
                self.save_image["v_value"] = None
            else:
                self.save_image["v_value"] = v_value
            self.effect_image = self.update(self.original_image)
            self.show_image(self.effect_image)

    def update_blur(self):
        k_value = self.uic.slider_blur.value()
        if self.original_image is not None:
            if k_value == -1:
                self.save_image["k_value"] = None
            else:
                self.save_image["k_value"] = k_value
            self.effect_image = self.update(self.original_image)
            self.show_image(self.effect_image)

    def update_saturation(self):
        s_value = self.uic.slider_saturation.value()
        if self.original_image is not None:
            if s_value == 0:
                self.save_image["s_value"] = 0
            else:
                self.save_image["s_value"] = s_value
            self.effect_image = self.update(self.original_image)
            self.show_image(self.effect_image)

    def update_old(self):
        i = self.uic.listView2.currentRow()
        if self.original_image is not None:
            self.save_image["old"] = i
            self.effect_image = self.update(self.original_image)
            self.show_image(self.effect_image)

    def reset_effect(self):
        self.save_image = {"filter": 0, "old": 0, "b_value": 1, "v_value": None, "k_value": None, "s_value": 0}
        self.uic.slider_brightness.setValue(50)
        self.uic.slider_blur.setValue(-1)
        self.uic.slider_vegnette.setValue(99)
        self.uic.slider_saturation.setValue(0)
        self.uic.listView.setCurrentRow(0)
        self.uic.listView2.setCurrentRow(0)
        self.effect_image = self.update(self.original_image)
        self.show_image(self.effect_image)

    def save_to(self):
        if self.original_image is not None:
            self.effect_image = self.update(self.original_image)
            save_path = QFileDialog.getSaveFileName(filter="*.jpg *.png *jpeg")
            print(save_path)
            cv2.imwrite(save_path[0].split(".")[0] + ".png", self.effect_image)

    def show(self):
        self.main_win.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
