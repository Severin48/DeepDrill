# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitled.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsView, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMenuBar,
    QPlainTextEdit, QPushButton, QSizePolicy, QSlider,
    QSplitter, QStatusBar, QVBoxLayout, QWidget)

class Ui_Comparator_window(object):
    def setupUi(self, Comparator_window):
        if not Comparator_window.objectName():
            Comparator_window.setObjectName(u"Comparator_window")
        Comparator_window.resize(1364, 899)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Comparator_window.sizePolicy().hasHeightForWidth())
        Comparator_window.setSizePolicy(sizePolicy)
        self.centralwidget = QWidget(Comparator_window)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalWidget = QWidget(self.centralwidget)
        self.verticalWidget.setObjectName(u"verticalWidget")
        self.verticalWidget.setEnabled(True)
        sizePolicy.setHeightForWidth(self.verticalWidget.sizePolicy().hasHeightForWidth())
        self.verticalWidget.setSizePolicy(sizePolicy)
        self.verticalWidget.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.horizontalLayout = QHBoxLayout(self.verticalWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.splitter_outer = QSplitter(self.verticalWidget)
        self.splitter_outer.setObjectName(u"splitter_outer")
        self.splitter_outer.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_outer.setHandleWidth(10)
        self.splitter_inner = QSplitter(self.splitter_outer)
        self.splitter_inner.setObjectName(u"splitter_inner")
        sizePolicy.setHeightForWidth(self.splitter_inner.sizePolicy().hasHeightForWidth())
        self.splitter_inner.setSizePolicy(sizePolicy)
        self.splitter_inner.setLineWidth(2)
        self.splitter_inner.setMidLineWidth(2)
        self.splitter_inner.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_inner.setHandleWidth(10)
        self.splitter_inner.setChildrenCollapsible(True)
        self.layoutWidget = QWidget(self.splitter_inner)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.layoutWidget)
        self.label.setObjectName(u"label")

        self.verticalLayout_2.addWidget(self.label)

        self.filter_code_left = QPlainTextEdit(self.layoutWidget)
        self.filter_code_left.setObjectName(u"filter_code_left")

        self.verticalLayout_2.addWidget(self.filter_code_left)

        self.apply_left_btn = QPushButton(self.layoutWidget)
        self.apply_left_btn.setObjectName(u"apply_left_btn")

        self.verticalLayout_2.addWidget(self.apply_left_btn)

        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 50)
        self.verticalLayout_2.setStretch(2, 1)
        self.splitter_inner.addWidget(self.layoutWidget)
        self.layoutWidget1 = QWidget(self.splitter_inner)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.verticalLayout = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.load_image2_btn = QPushButton(self.layoutWidget1)
        self.load_image2_btn.setObjectName(u"load_image2_btn")

        self.horizontalLayout_4.addWidget(self.load_image2_btn)

        self.load_image1_btn = QPushButton(self.layoutWidget1)
        self.load_image1_btn.setObjectName(u"load_image1_btn")

        self.horizontalLayout_4.addWidget(self.load_image1_btn)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.outer_frame = QFrame(self.layoutWidget1)
        self.outer_frame.setObjectName(u"outer_frame")
        self.outer_frame.setEnabled(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.outer_frame.sizePolicy().hasHeightForWidth())
        self.outer_frame.setSizePolicy(sizePolicy1)
        self.outer_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.outer_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.outer_frame.setLineWidth(0)
        self.right_frame = QFrame(self.outer_frame)
        self.right_frame.setObjectName(u"right_frame")
        self.right_frame.setGeometry(QRect(0, 0, 276, 212))
        sizePolicy.setHeightForWidth(self.right_frame.sizePolicy().hasHeightForWidth())
        self.right_frame.setSizePolicy(sizePolicy)
        self.right_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.right_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.right_frame)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.graphicsView_right = QGraphicsView(self.right_frame)
        self.graphicsView_right.setObjectName(u"graphicsView_right")

        self.horizontalLayout_3.addWidget(self.graphicsView_right)

        self.left_frame = QFrame(self.outer_frame)
        self.left_frame.setObjectName(u"left_frame")
        self.left_frame.setGeometry(QRect(0, 0, 276, 212))
        sizePolicy.setHeightForWidth(self.left_frame.sizePolicy().hasHeightForWidth())
        self.left_frame.setSizePolicy(sizePolicy)
        self.left_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.left_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.left_frame)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.graphicsView_left = QGraphicsView(self.left_frame)
        self.graphicsView_left.setObjectName(u"graphicsView_left")

        self.horizontalLayout_2.addWidget(self.graphicsView_left)


        self.verticalLayout.addWidget(self.outer_frame)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.scale_slider = QSlider(self.layoutWidget1)
        self.scale_slider.setObjectName(u"scale_slider")
        self.scale_slider.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayout_5.addWidget(self.scale_slider)


        self.verticalLayout.addLayout(self.verticalLayout_5)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 15)
        self.verticalLayout.setStretch(2, 1)
        self.splitter_inner.addWidget(self.layoutWidget1)
        self.splitter_outer.addWidget(self.splitter_inner)
        self.verticalLayoutWidget = QWidget(self.splitter_outer)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayout_3 = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_4 = QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName(u"label_4")

        self.verticalLayout_3.addWidget(self.label_4)

        self.filter_code_right = QPlainTextEdit(self.verticalLayoutWidget)
        self.filter_code_right.setObjectName(u"filter_code_right")

        self.verticalLayout_3.addWidget(self.filter_code_right)

        self.apply_right_btn = QPushButton(self.verticalLayoutWidget)
        self.apply_right_btn.setObjectName(u"apply_right_btn")

        self.verticalLayout_3.addWidget(self.apply_right_btn)

        self.splitter_outer.addWidget(self.verticalLayoutWidget)

        self.horizontalLayout.addWidget(self.splitter_outer)


        self.gridLayout.addWidget(self.verticalWidget, 0, 0, 1, 1)

        Comparator_window.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Comparator_window)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1364, 19))
        Comparator_window.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Comparator_window)
        self.statusbar.setObjectName(u"statusbar")
        Comparator_window.setStatusBar(self.statusbar)

        self.retranslateUi(Comparator_window)

        QMetaObject.connectSlotsByName(Comparator_window)
    # setupUi

    def retranslateUi(self, Comparator_window):
        Comparator_window.setWindowTitle(QCoreApplication.translate("Comparator_window", u"Comparator", None))
        self.label.setText(QCoreApplication.translate("Comparator_window", u"Filter Left", None))
        self.apply_left_btn.setText(QCoreApplication.translate("Comparator_window", u"Apply", None))
        self.load_image2_btn.setText(QCoreApplication.translate("Comparator_window", u"Load Left", None))
        self.load_image1_btn.setText(QCoreApplication.translate("Comparator_window", u"Load Right", None))
        self.label_4.setText(QCoreApplication.translate("Comparator_window", u"Filter Right", None))
        self.apply_right_btn.setText(QCoreApplication.translate("Comparator_window", u"Apply", None))
    # retranslateUi

