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
        Comparator_window.resize(919, 720)
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
        self.splitter = QSplitter(self.verticalWidget)
        self.splitter.setObjectName(u"splitter")
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setLineWidth(2)
        self.splitter.setMidLineWidth(2)
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(True)
        self.widget = QWidget(self.splitter)
        self.widget.setObjectName(u"widget")
        self.verticalLayout_2 = QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")

        self.verticalLayout_2.addWidget(self.label)

        self.filter_code = QPlainTextEdit(self.widget)
        self.filter_code.setObjectName(u"filter_code")

        self.verticalLayout_2.addWidget(self.filter_code)

        self.apply_filter_btn = QPushButton(self.widget)
        self.apply_filter_btn.setObjectName(u"apply_filter_btn")

        self.verticalLayout_2.addWidget(self.apply_filter_btn)

        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 50)
        self.verticalLayout_2.setStretch(2, 1)
        self.splitter.addWidget(self.widget)
        self.widget1 = QWidget(self.splitter)
        self.widget1.setObjectName(u"widget1")
        self.verticalLayout = QVBoxLayout(self.widget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.load_image2_btn = QPushButton(self.widget1)
        self.load_image2_btn.setObjectName(u"load_image2_btn")

        self.horizontalLayout_4.addWidget(self.load_image2_btn)

        self.label_2 = QLabel(self.widget1)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_4.addWidget(self.label_2)

        self.load_image1_btn = QPushButton(self.widget1)
        self.load_image1_btn.setObjectName(u"load_image1_btn")

        self.horizontalLayout_4.addWidget(self.load_image1_btn)

        self.label_3 = QLabel(self.widget1)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_4.addWidget(self.label_3)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.outer_frame = QFrame(self.widget1)
        self.outer_frame.setObjectName(u"outer_frame")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.outer_frame.sizePolicy().hasHeightForWidth())
        self.outer_frame.setSizePolicy(sizePolicy1)
        self.outer_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.outer_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.compare_frame = QFrame(self.outer_frame)
        self.compare_frame.setObjectName(u"compare_frame")
        self.compare_frame.setGeometry(QRect(0, 0, 300, 300))
        sizePolicy.setHeightForWidth(self.compare_frame.sizePolicy().hasHeightForWidth())
        self.compare_frame.setSizePolicy(sizePolicy)
        self.compare_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.compare_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.compare_frame)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.graphicsView_compare = QGraphicsView(self.compare_frame)
        self.graphicsView_compare.setObjectName(u"graphicsView_compare")

        self.horizontalLayout_6.addWidget(self.graphicsView_compare)

        self.original_frame = QFrame(self.outer_frame)
        self.original_frame.setObjectName(u"original_frame")
        self.original_frame.setGeometry(QRect(0, 0, 300, 300))
        sizePolicy.setHeightForWidth(self.original_frame.sizePolicy().hasHeightForWidth())
        self.original_frame.setSizePolicy(sizePolicy)
        self.original_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.original_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.original_frame.setLineWidth(-5)
        self.horizontalLayout_5 = QHBoxLayout(self.original_frame)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.graphicsView_original = QGraphicsView(self.original_frame)
        self.graphicsView_original.setObjectName(u"graphicsView_original")

        self.horizontalLayout_5.addWidget(self.graphicsView_original)


        self.verticalLayout.addWidget(self.outer_frame)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.scale_slider = QSlider(self.widget1)
        self.scale_slider.setObjectName(u"scale_slider")
        self.scale_slider.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayout_5.addWidget(self.scale_slider)


        self.verticalLayout.addLayout(self.verticalLayout_5)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 15)
        self.verticalLayout.setStretch(2, 1)
        self.splitter.addWidget(self.widget1)

        self.horizontalLayout.addWidget(self.splitter)


        self.gridLayout.addWidget(self.verticalWidget, 0, 0, 1, 1)

        Comparator_window.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Comparator_window)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 919, 19))
        Comparator_window.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Comparator_window)
        self.statusbar.setObjectName(u"statusbar")
        Comparator_window.setStatusBar(self.statusbar)

        self.retranslateUi(Comparator_window)

        QMetaObject.connectSlotsByName(Comparator_window)
    # setupUi

    def retranslateUi(self, Comparator_window):
        Comparator_window.setWindowTitle(QCoreApplication.translate("Comparator_window", u"Comparator", None))
        self.label.setText(QCoreApplication.translate("Comparator_window", u"Filter", None))
        self.apply_filter_btn.setText(QCoreApplication.translate("Comparator_window", u"Apply", None))
        self.load_image2_btn.setText(QCoreApplication.translate("Comparator_window", u"Load", None))
        self.label_2.setText(QCoreApplication.translate("Comparator_window", u"Image 1", None))
        self.load_image1_btn.setText(QCoreApplication.translate("Comparator_window", u"Load", None))
        self.label_3.setText(QCoreApplication.translate("Comparator_window", u"Image 2", None))
    # retranslateUi

