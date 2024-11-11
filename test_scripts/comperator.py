from sys import exception

from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFileDialog, QFrame
from PySide6.QtGui import QPixmap, QImage,QPainter
from PySide6.QtCore import Qt, QRect, QPoint, QSignalBlocker
import PySide6.QtGui
from prettytable import PrettyTable
from tqdm import tqdm
from window10 import Ui_Comparator_window 
import cv2
import numpy as np
import image_similarity_measures.evaluate as img_eval

max_zoom = 300

class ResizableFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        
        self.grip_size = 50
        self.is_resizing = False
        self.resize_direction = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPosition().toPoint()
            self.original_rect = self.geometry()
            
            if self.is_on_left_edge(event.position()):
                self.resize_direction = 'left'
            elif self.is_on_right_edge(event.position()):
                self.resize_direction = 'right'
            else:
                self.resize_direction = None

            self.is_resizing = self.resize_direction is not None

    def mouseMoveEvent(self, event):
        if not self.is_resizing:
            if self.is_on_left_edge(event.position()) or self.is_on_right_edge(event.position()):
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        else:
            delta = event.globalPosition().toPoint() - self.start_pos
            rect = self.original_rect
            if self.resize_direction == 'right':
                new_rect = QRect(rect.x(), rect.y(), rect.width() + delta.x(), rect.height())

            self.setGeometry(new_rect)

    def mouseReleaseEvent(self, event):
        self.is_resizing = False
        self.setCursor(Qt.ArrowCursor)

    def is_on_left_edge(self, pos):
        return pos.x() < self.grip_size

    def is_on_right_edge(self, pos):
        return pos.x() > self.width() - self.grip_size

def calc_metrics(metrics, original, compare):
    results = PrettyTable(["Metric", "Value"])

    for metric in tqdm(metrics):
        metric_func = img_eval.metric_functions[metric]

        metric_value = float(metric_func(original, compare))
        results.add_row([metric,round(metric_value,4)])

    return results

def cv2_to_qpixmap(cv_image):
    """Convert an OpenCV image (BGR format) to QPixmap."""
    height, width, channels = cv_image.shape
    bytes_per_line = channels * width
    q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    return QPixmap.fromImage(q_image)

class Comperator(QMainWindow, Ui_Comparator_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.original_image_left = None
        self.original_image_right = None

        self.image_right = None
        self.image_left = None
        
        self.zoom_level = 0

        self.custom_elements()
        self.connect_ui()

    def finalize_ui(self):
        # splitter size 
        self.splitter_outer.setSizes([1,8])
        self.splitter_inner.setSizes([8,1])


        self.left_frame.setGeometry(0,0,self.outer_frame.width()/2,self.outer_frame.height())
        self.right_frame.setGeometry(0,0,self.outer_frame.width(),self.outer_frame.height())


        # if DEBUG
        self.load_image('test_img/original/seahorse1_1920x1080.png',True)
        self.load_image('test_img/original/seahorse1_4k.jpg',False)

        filter_right ='''
# Scale
#image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)

# sharpen
sigma = 1.0
amount = 0.15
blur = cv2.GaussianBlur(image, (0, 0), sigma)
image = cv2.addWeighted(image, 1 + amount, blur , -amount, 0)
'''
        self.filter_code_right.setPlainText(filter_right)
        # endif DEBUG

        self.apply_filter_right()

    def custom_elements(self):
        geo = self.left_frame.geometry()
        children = self.left_frame.children()
        self.left_frame.setStyleSheet("border:0px;opacity:0%")
        
        new_frame = ResizableFrame(self.left_frame.parent())
        new_frame.setGeometry(geo.x(),geo.y(),geo.width(),geo.height())
        new_frame.setLayout(self.left_frame.layout())

        # copy children
        for child in children:
            child.setParent(new_frame)

        self.left_frame= new_frame

        self.left_frame.setStyleSheet("border-right:3px solid;border-style: solid;border-color: rgb(10, 10, 10)")
        self.right_frame.setStyleSheet("border-right:3px solid;border-style: solid;border-color: rgb(0, 0, )")

        self.graphicsView_right.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);
        self.graphicsView_right.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff);
        self.graphicsView_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);
        self.graphicsView_left.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff);

        self.graphicsView_left.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.graphicsView_left.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.graphicsView_left.setRenderHint(QPainter.Antialiasing, False)

        self.graphicsView_right.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.graphicsView_right.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.graphicsView_right.setRenderHint(QPainter.Antialiasing, False)


    def resizeEvent(self,event):
        self.left_frame.setGeometry(0,0,self.outer_frame.width()/2,self.outer_frame.height())
        self.right_frame.setGeometry(0,0,self.outer_frame.width(),self.outer_frame.height())
        QMainWindow.resizeEvent(self, event)

    def connect_ui(self):
        self.load_image1_btn.clicked.connect(self.open_compare_file_dialog)
        self.load_image2_btn.clicked.connect(self.open_original_file_dialog)

        self.scale_slider.setMinimum(0)
        self.scale_slider.setMaximum(max_zoom)
        self.scale_slider.valueChanged.connect(self.slider_changed)

        # Connect horizontal scrollbars
        self.graphicsView_left.horizontalScrollBar().valueChanged.connect(self.graphicsView_right.horizontalScrollBar().setValue)
        self.graphicsView_right.horizontalScrollBar().valueChanged.connect(self.graphicsView_left.horizontalScrollBar().setValue)

        # Connect vertical scrollbars
        self.graphicsView_left.verticalScrollBar().valueChanged.connect(self.graphicsView_right.verticalScrollBar().setValue)
        self.graphicsView_right.verticalScrollBar().valueChanged.connect(self.graphicsView_left.verticalScrollBar().setValue)

        # connect filter button
        self.apply_left_btn.clicked.connect(self.apply_filter_left)
        self.apply_right_btn.clicked.connect(self.apply_filter_right)

        # connect splitter
        self.splitter_inner.splitterMoved.connect(self.splitter_moved)
        self.splitter_outer.splitterMoved.connect(self.splitter_moved)

    # ########## UI Connections ##########
    def open_original_file_dialog(self):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open original Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)")
            
            if file_path:
                self.load_image(file_path, True)

    def splitter_moved(self):
        self.left_frame.setGeometry(0,0,self.outer_frame.width()/2,self.outer_frame.height())
        self.right_frame.setGeometry(0,0,self.outer_frame.width(),self.outer_frame.height())

    def open_compare_file_dialog(self):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open compare Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)")
            
            if file_path:
                self.load_image(file_path, False)

    def slider_changed(self):
        before_zoom = self.zoom_level
        self.zoom_level = self.scale_slider.value()
        zoom_factor_small = 1.01
        zoom_factor_big= 1.01

        if before_zoom > self.zoom_level:
            zoom_factor_small = 1/zoom_factor_small
            zoom_factor_big = 1/zoom_factor_big


        self.graphicsView_left.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.graphicsView_left.scale(zoom_factor_big,zoom_factor_big)
        self.graphicsView_right.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.graphicsView_right.scale(zoom_factor_small, zoom_factor_small)


        
    def load_image(self, image_path, isLeft):
        cv_image = cv2.imread(image_path)
        
        if isLeft:
            self.original_image_left = cv_image
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.scene_left = QGraphicsScene()
            pixmap_left = cv2_to_qpixmap(cv_image)
            self.image_item_left = QGraphicsPixmapItem(pixmap_left)
            self.scene_left.addItem(self.image_item_left)
            self.graphicsView_left.setScene(self.scene_left)
            self.graphicsView_left.update()
            self.graphicsView_left.setDragMode(QGraphicsView.ScrollHandDrag)
            self.image_left = cv_image
        else:
            self.original_image_right = cv_image
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.scene_right = QGraphicsScene()
            pixmap_right = cv2_to_qpixmap(cv_image)
            self.image_item_right = QGraphicsPixmapItem(pixmap_right)
            self.scene_right.addItem(self.image_item_right)
            self.graphicsView_right.setScene(self.scene_right)
            self.graphicsView_right.setDragMode(QGraphicsView.ScrollHandDrag)
            self.image_right= cv_image

            
    def apply_filter_right(self):
        try:
            image = self.original_image_right.copy()

            # highly problematic, but we are professionals...
            filter = self.filter_code_right.toPlainText()
            local_scope = {"image":image}
            exec(filter, globals(),local_scope)
            image = local_scope["image"]

            cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.scene_right = QGraphicsScene()
            pixmap_right = cv2_to_qpixmap(cv_image)
            self.image_item_right = QGraphicsPixmapItem(pixmap_right)
            self.scene_right.addItem(self.image_item_right)
            self.graphicsView_right.setScene(self.scene_right)
            self.graphicsView_right.setDragMode(QGraphicsView.ScrollHandDrag)
            self.graphicsView_right.update()

            self.image_right= image

            # also calculate metrics on new image
            metrics = ["psnr",'ssim']
            print(calc_metrics(metrics,self.original_image_left, image))
        except Exception as e:
            print("Exception occurred when trying to apply filter: ", e)


    def apply_filter_left(self):
        try:
            image = self.original_image_left.copy()

            # highly problematic, but we are professionals...
            filter = self.filter_code_left.toPlainText()
            local_scope = {"image":image}
            exec(filter, globals(),local_scope)
            image = local_scope["image"]

            cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.scene_left = QGraphicsScene()
            pixmap_left = cv2_to_qpixmap(cv_image)
            self.image_item_left = QGraphicsPixmapItem(pixmap_left)
            self.scene_left.addItem(self.image_item_left)
            self.graphicsView_left.setScene(self.scene_left)
            self.graphicsView_left.setDragMode(QGraphicsView.ScrollHandDrag)
            self.graphicsView_left.update()

            self.image_left = image

            # also calculate metrics on new image
            metrics = ["psnr",'ssim']
            print(calc_metrics(metrics,self.image_left, self.image_right))
        except Exception as e:
            print("Exception occurred when trying to apply filter: ", e)

## END COMPONENT ##


app = QApplication()
comperator = Comperator()
comperator.show()
comperator.finalize_ui()
app.exec()