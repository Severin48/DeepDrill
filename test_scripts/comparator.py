#from sys import exception

from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFileDialog, QFrame
from PySide6.QtGui import QPixmap, QImage,QPainter
from PySide6.QtCore import Qt, QRect, QPoint, QSignalBlocker
import PySide6.QtGui
from prettytable import PrettyTable
from tqdm import tqdm
from window10 import Ui_Comparator_window 
import cv2
import numpy as np
import numpy as np
from tqdm import tqdm
from numba import jit, prange
import numpy as np


# ###### global functions ######
def cv2_to_qpixmap(cv_image):
    """Convert an OpenCV image (BGR format) to QPixmap (RGB)."""
    height, width, channels = cv_image.shape
    bytes_per_line = channels * width
    q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    return QPixmap.fromImage(q_image)

# ###### Classes ######

class ResizableFrame(QFrame):
    '''
    QFrame with the ability to be resized by dragging the edge
    '''
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

class Comparator(QMainWindow, Ui_Comparator_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.original_image_left = None
        self.original_image_right = None

        self.image_right = None
        self.image_left = None
        
        self.zoom_level = 0

        self.create_custom_elements()
        self.connect_ui()

    def finalize_ui(self):
        # splitter size 
        self.splitter_outer.setSizes([1,8])
        self.splitter_inner.setSizes([8,1])

        self.left_frame.setGeometry(0,0,self.outer_frame.width()/2,self.outer_frame.height())
        self.right_frame.setGeometry(0,0,self.outer_frame.width(),self.outer_frame.height())

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

# if DEBUG
        self.load_image('test_img/original/seahorse1_1920x1080.png',True)
        self.load_image('test_img/original/seahorse1_4k.jpg',False)

        filter_right ='''

# Downscaler options:
# INTER_NEAREST, INTER_LINEAR, INTER_AREA,
# INTER_CUBIC, INTER_LANCZOS4

#### Scale ####
if 0:
    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
else:
    image = scale(image, (1920, 1080), 3)

#### Sharpen ####
if 0:
    sigma = 1.0
    amount = 0.15
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    image = cv2.addWeighted(image, 1 + amount, blur , -amount, 0)

#### Show diff with Pink pixels ####
if 0:
    # options: 
    thresh = 60
    thick = False  

    diff = cv2.absdiff(left_image, image)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_thresh = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
    mask = diff_thresh > 0
    pink = [255, 0, 255]

    if thick:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    image[mask == 1] = pink

'''
        self.filter_code_right.setPlainText(filter_right)
        self.apply_filter_right()
        self.slider_changed()

        # endif DEBUG

    def create_custom_elements(self):
        geo = self.left_frame.geometry()
        children = self.left_frame.children()
        self.left_frame.setStyleSheet("border:0px;opacity:0%")
        
        # create custom Frame and copy all relevant attributes from placeholder Frame created by the UI-Designer
        new_frame = ResizableFrame(self.left_frame.parent())
        new_frame.setGeometry(geo.x(),geo.y(),geo.width(),geo.height())
        new_frame.setLayout(self.left_frame.layout())

        for child in children:
            child.setParent(new_frame)

        self.left_frame= new_frame

        self.left_frame.setStyleSheet("border-right:3px solid;border-style: solid;border-color: rgb(10, 10, 10)")
        self.right_frame.setStyleSheet("border-right:3px solid;border-style: solid;border-color: rgb(0, 0, )")

    def connect_ui(self):
        self.load_image1_btn.clicked.connect(self.open_right_file_dialog)
        self.load_image2_btn.clicked.connect(self.open_left_file_dialog)

        self.scale_slider.setMinimum(0)
        self.scale_slider.setMaximum(500)
        self.scale_slider.valueChanged.connect(self.slider_changed)

        # Connect horizontal scrollbars/
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

    def resizeEvent(self,event):
        self.left_frame.setGeometry(0,0,self.outer_frame.width()/2,self.outer_frame.height())
        self.right_frame.setGeometry(0,0,self.outer_frame.width(),self.outer_frame.height())
        QMainWindow.resizeEvent(self, event)

    # ########## UI Event Connections ##########
    def open_left_file_dialog(self):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open original Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)")
            
            if file_path:
                self.load_image(file_path, True)

    def open_right_file_dialog(self):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open compare Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)")
            
            if file_path:
                self.load_image(file_path, False)

    def splitter_moved(self):
        '''Adjusts the left and right frame to take half and the full size of the parant'''
        self.left_frame.setGeometry(0,0,self.outer_frame.width()/2,self.outer_frame.height())
        self.right_frame.setGeometry(0,0,self.outer_frame.width(),self.outer_frame.height())

    def slider_changed(self):
        '''Changes the zoom factor based on the sliders movement'''
        before_zoom = self.zoom_level
        self.zoom_level = self.scale_slider.value()
        zoom_factor= 1.01

        if before_zoom > self.zoom_level:
            zoom_factor = 1/zoom_factor

        zoom = zoom_factor * ((self.zoom_level / 100) +0.5)

        # block signals while zooming to disable unpredictable effects on the scrollbar positions
        self.graphicsView_right.horizontalScrollBar().blockSignals(True)
        self.graphicsView_right.verticalScrollBar().blockSignals(True)

        self.graphicsView_left.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.graphicsView_left.resetTransform()
        self.graphicsView_left.scale(zoom,zoom)

        # and unblock so that the scrollbars are synced to the right Frame
        self.graphicsView_right.horizontalScrollBar().blockSignals(False)
        self.graphicsView_right.verticalScrollBar().blockSignals(False)

        self.graphicsView_right.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.graphicsView_right.resetTransform()
        self.graphicsView_right.scale(zoom, zoom)


        
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
        # evaluate filter
        image = self.original_image_right.copy()
        filter = self.filter_code_right.toPlainText()
        local_scope = {"image":image, "left_image":self.original_image_left}
        # CRITICAL: HERE PYTHON CODE FROM AN FREE TEXT INPUT IS EVALUATED WITHOUT ANY CHECKS!
        exec(filter, globals(),local_scope)
        image = local_scope["image"]

        # update UI 
        cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.scene_right = QGraphicsScene()
        pixmap_right = cv2_to_qpixmap(cv_image)
        self.image_item_right = QGraphicsPixmapItem(pixmap_right)
        self.scene_right.addItem(self.image_item_right)
        self.graphicsView_right.setScene(self.scene_right)
        self.graphicsView_right.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView_right.update()

        self.image_right= image


    def apply_filter_left(self):
        try:
            # evaluate filter
            image = self.original_image_left.copy()
            filter = self.filter_code_left.toPlainText()
            local_scope = {"image":image}
            # CRITICAL: HERE PYTHON CODE FROM AN FREE TEXT INPUT IS EVALUATED WITHOUT ANY CHECKS!
            exec(filter, globals(),local_scope)
            image = local_scope["image"]

            # update UI 
            cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.scene_left = QGraphicsScene()
            pixmap_left = cv2_to_qpixmap(cv_image)
            self.image_item_left = QGraphicsPixmapItem(pixmap_left)
            self.scene_left.addItem(self.image_item_left)
            self.graphicsView_left.setScene(self.scene_left)
            self.graphicsView_left.setDragMode(QGraphicsView.ScrollHandDrag)
            self.graphicsView_left.update()

            self.image_left = image

        except Exception as e:
            print("Exception occurred when trying to apply filter: ", e)
## END COMPONENT ##


# only usable with numba and supported numpy version
# else its to slow ...
@jit(nopython=True, parallel=True)
def scale(image, size, a=3):
    def lanczos_kernel(a, size):
        kernel = np.zeros(size)
        for i in range(size):
            x = i - (size - 1) / 2.0
            if x == 0:
                kernel[i] = 1.0
            else:
                x = np.pi * x
                kernel[i] = a * np.sin(x) * np.sin(x / a) / (x ** 2)
        
        kernel_sum = np.sum(kernel)
        kernel /= kernel_sum

        return kernel
    
    original_height, original_width = image.shape[0], image.shape[1]
    new_width, new_height  = size

    kernel_size = 2 * a
    kernel = lanczos_kernel(a, kernel_size)

    scaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            orig_x = (x + 0.5) * original_width / new_width - 0.5
            orig_y = (y + 0.5) * original_height / new_height - 0.5

            r, g, b = 0.0, 0.0, 0.0
            for ky in range(-a+1, a):
                for kx in range(-a+1, a):
                    iy = int(orig_y) + ky
                    ix = int(orig_x) + kx
                    
                    if 0 <= iy < original_height and 0 <= ix < original_width:
                        weight = kernel[ky + a - 1] * kernel[kx + a - 1]
                        r += image[iy, ix, 0] * weight
                        g += image[iy, ix, 1] * weight
                        b += image[iy, ix, 2] * weight
                        
            scaled_image[y, x] = [min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b))]

    return scaled_image

app = QApplication()
comparator = Comparator()
comparator.show()
comparator.finalize_ui()
app.exec()