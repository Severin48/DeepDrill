from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFileDialog, QFrame
from PySide6.QtGui import QPixmap, QImage,QPainter
from PySide6.QtCore import Qt, QRect, QPoint
import PySide6.QtGui
from prettytable import PrettyTable
from tqdm import tqdm
from window7 import Ui_Comparator_window 
import cv2
import numpy as np
import image_similarity_measures.evaluate as img_eval

max_zoom = 200

class ResizableFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.StyledPanel)
        
        self.grip_size = 30 
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
            elif self.is_on_top_edge(event.position()) or self.is_on_bottom_edge(event.position()):
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        else:
            delta = event.globalPosition().toPoint() - self.start_pos
            rect = self.original_rect
            if self.resize_direction == 'left':
                new_rect = QRect(rect.x() + delta.x(), rect.y(), rect.width() - delta.x(), rect.height())
            elif self.resize_direction == 'right':
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

        self.original_image = None
        self.compare_image = None
        self.zoom_level = 0
        self.filter = '' 

        self.custom_elements()
        self.connect_ui()

    def finalize_ui(self):
        # splitter size 
        self.splitter.setSizes([1,8])

        self.original_frame.setGeometry(0,0,self.outer_frame.width()/2,self.outer_frame.height())
        self.compare_frame.setGeometry(0,0,self.outer_frame.width(),self.outer_frame.height())


        # if DEBUG
        self.load_image('test_img/original/seahorse1_4k.jpg',False)
        self.load_image('test_img/original/seahorse1_1920x1080.png',True)

        self.filter='''
# Scale
#image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)

# sharpen
sigma = 1.0
amount = 0.15
blur = cv2.GaussianBlur(image, (0, 0), sigma)
image = cv2.addWeighted(image, 1 + amount, blur , -amount, 0)
'''
        self.filter_code.setPlainText(self.filter)
        # endif DEBUG
        

    def custom_elements(self):
        geo = self.original_frame.geometry()
        children = self.original_frame.children()
        self.original_frame.setStyleSheet("border:0px;opacity:0%")
        
        new_frame = ResizableFrame(self.original_frame.parent())
        new_frame.setGeometry(geo.x(),geo.y(),geo.width(),geo.height())
        new_frame.setLayout(self.original_frame.layout())

        # copy children
        for child in children:
            child.setParent(new_frame)

        self.original_frame = new_frame

        self.original_frame.setStyleSheet("border-right:3px solid;border-style: solid;border-color: rgb(10, 10, 10)")
        self.compare_frame.setStyleSheet("border-right:3px solid;border-style: solid;border-color: rgb(0, 0, )")

        self.graphicsView_compare.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);
        self.graphicsView_compare.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff);
        self.graphicsView_original.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);
        self.graphicsView_original.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff);

        self.graphicsView_original.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.graphicsView_original.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.graphicsView_original.setRenderHint(QPainter.Antialiasing, False)

        self.graphicsView_compare.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.graphicsView_compare.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.graphicsView_compare.setRenderHint(QPainter.Antialiasing, False)


    def resizeEvent(self,event):
        self.original_frame.setGeometry(0,0,self.outer_frame.width()/2,self.outer_frame.height())
        self.compare_frame.setGeometry(0,0,self.outer_frame.width(),self.outer_frame.height())
        QMainWindow.resizeEvent(self, event)

    def connect_ui(self):
        self.load_image1_btn.clicked.connect(self.open_compare_file_dialog)
        self.load_image2_btn.clicked.connect(self.open_original_file_dialog)

        self.scale_slider.setMinimum(0)
        self.scale_slider.setMaximum(max_zoom)
        self.scale_slider.valueChanged.connect(self.slider_changed)

        # Connect horizontal scrollbars
        self.graphicsView_original.horizontalScrollBar().valueChanged.connect(self.graphicsView_compare.horizontalScrollBar().setValue)
        self.graphicsView_compare.horizontalScrollBar().valueChanged.connect(self.graphicsView_original.horizontalScrollBar().setValue)

        # Connect vertical scrollbars
        self.graphicsView_original.verticalScrollBar().valueChanged.connect(self.graphicsView_compare.verticalScrollBar().setValue)
        self.graphicsView_compare.verticalScrollBar().valueChanged.connect(self.graphicsView_original.verticalScrollBar().setValue)

        # connect filter button
        self.apply_filter_btn.clicked.connect(self.apply_filter)

        # connect splitter
        self.splitter.splitterMoved.connect(self.splitter_moved)


    # ########## UI Connections ##########
    def open_original_file_dialog(self):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open original Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)")
            
            if file_path:
                self.load_image(file_path, True)

    def splitter_moved(self):
        self.original_frame.setGeometry(0,0,self.outer_frame.width()/2,self.outer_frame.height())
        self.compare_frame.setGeometry(0,0,self.outer_frame.width(),self.outer_frame.height())

    def open_compare_file_dialog(self):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open compare Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)")
            
            if file_path:
                self.load_image(file_path, False)

    def slider_changed(self):
        before_zoom = self.zoom_level
        self.zoom_level = self.scale_slider.value()
        zoom_factor = 1.01

        if before_zoom > self.zoom_level:
            zoom_factor = 1/zoom_factor

        ## Zoom for self.graphicsView_original
        self.graphicsView_original.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.graphicsView_original.scale(zoom_factor, zoom_factor)
        self.graphicsView_compare.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.graphicsView_compare.scale(zoom_factor, zoom_factor)

        #self.graphicsView_compare.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        #self.graphicsView_compare.scale(zoom_factor, zoom_factor)






        
    def load_image(self, image_path, isOriginal):
        cv_image = cv2.imread(image_path)
        
        if isOriginal:
            self.original_image = cv_image
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.scene_original = QGraphicsScene()
            pixmap_original = cv2_to_qpixmap(cv_image)
            self.image_item_original = QGraphicsPixmapItem(pixmap_original)
            self.scene_original.addItem(self.image_item_original)
            self.graphicsView_original.setScene(self.scene_original)
            self.graphicsView_original.update()
            self.graphicsView_original.setDragMode(QGraphicsView.ScrollHandDrag)
            #self.graphicsView_original.fitInView(self.scene_original.sceneRect(), Qt.KeepAspectRatio)
        else:
            self.compare_image_original = cv_image
            self.compare_image = cv_image
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.scene_compare = QGraphicsScene()
            pixmap_compare = cv2_to_qpixmap(cv_image)
            self.image_item_compare = QGraphicsPixmapItem(pixmap_compare)
            self.scene_compare.addItem(self.image_item_compare)
            self.graphicsView_compare.setScene(self.scene_compare)
            self.graphicsView_compare.setDragMode(QGraphicsView.ScrollHandDrag)
            #self.graphicsView_compare.fitInView(self.scene_compare.sceneRect(), Qt.KeepAspectRatio)
        
    def apply_filter(self):
        image = self.compare_image_original.copy()

        # highly problematic, but we are professionals... 
        filter = self.filter_code.toPlainText()
        local_scope = {"image":image}
        exec(filter, globals(),local_scope)
        image = local_scope["image"]

        cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.scene_compare = QGraphicsScene()
        pixmap_compare = cv2_to_qpixmap(cv_image)
        self.image_item_compare = QGraphicsPixmapItem(pixmap_compare)
        self.scene_compare.addItem(self.image_item_compare)
        self.graphicsView_compare.setScene(self.scene_compare)
        self.graphicsView_compare.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView_compare.update()

        # also calculate metrics on new image
        metrics = ["psnr",'ssim']
        print(calc_metrics(metrics,self.original_image, image))


## END COMPONENT ##


app = QApplication()
comperator = Comperator()
comperator.show()
comperator.finalize_ui()
app.exec()