from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QGridLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image

from gui.graphics_view import GraphicsView


class SegmentationView(QWidget):
    def __init__(self):
        super().__init__()
        #self.shared_data = shared_data
        self.setupUI()

    def setupUI(self):
        # Grid layout
        self.view_layout = QGridLayout()

        self.setupInputGraphicView()
        self.setupOutputGraphicView()

        # Buttons
        self.loadImageButton = QPushButton("Загрузить изображение")
        self.processImageButton = QPushButton("Выделить маску изображения")
        self.saveImageButton = QPushButton("Сохранить маску")

        # Connecting buttons to functions
        self.loadImageButton.clicked.connect(self.loadImage)
        self.processImageButton.clicked.connect(self.processImage)
        self.saveImageButton.clicked.connect(self.saveImage)



        self.view_layout.addWidget(self.loadImageButton, 0, 0, 1 , 1)  # Row 0, Column 0
        self.view_layout.addWidget(self.processImageButton, 0, 1, 1, 1)  # Row 0, Column 1
        self.view_layout.addWidget(self.saveImageButton, 0, 2, 1 , 1)  # Row 1, Column 2

        # Set the layout to the widget
        self.setLayout(self.view_layout)

    def setupInputGraphicView(self):
        self.inputGraphicsView = GraphicsView(QSize(600,400),self)
        self.view_layout.addWidget(self.inputGraphicsView, 1, 0, -1, 1)  # Row 0, Column 0

    def setupOutputGraphicView(self):
        self.outputGraphicsView = GraphicsView(QSize(600,400),self)
        self.view_layout.addWidget(self.outputGraphicsView, 1, 2,-1,1)  # Row 0, Column 2

    def loadImage(self):
        # Open a dialog to select an image file
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                  "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif)")

        if filePath:
            self.inImage = Image.open(filePath)
            self.inImage = self.inImage.convert("RGB")  # Convert to RGB

            # Convert PIL image to QImage
            data = self.inImage.tobytes("raw", "RGB")
            qImage = QImage(data, self.inImage.size[0], self.inImage.size[1], QImage.Format_RGB888)
            self.inputGraphicsView.set_image(QPixmap.fromImage(qImage))

            # Store the file path in shared data if needed
            #self.shared_data.inputImagePath = filePath

    def processImage(self):
        # Logic to process the image,
        
        pass

    def saveImage(self):
        # Open a dialog to get the file name to save the image
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)")

        #if filePath:
            # Save the processed image to the specified path
            # Assuming the processed image is stored in shared_data or a class variable
            #self.shared_data.processedImage.save(filePath)