
from PyQt5.QtCore import Qt, QFile, QTextStream
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QGridLayout, QMainWindow, QStackedWidget, \
    QApplication, QRadioButton, QGroupBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import sys
from qt_material import apply_stylesheet

from segmentation_view import SegmentationView
from approximation_view import ApproximationView
from optimization_view import OptimizationView

class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.shared_data = SharedData()

        self.setWindowTitle('PHOTONICA')
        #self.load_styles()

        self.grid_layout = QGridLayout()
        self.central_widget = QWidget()


        self.stacked_widget = QStackedWidget()

        # Initialize views
        self.segmentation_view = SegmentationView()
        self.approximation_view = ApproximationView()
        self.optimization_view = OptimizationView()

        # Add views to stacked widget
        self.stacked_widget.addWidget(self.segmentation_view)
        self.stacked_widget.addWidget(self.approximation_view)
        self.stacked_widget.addWidget(self.optimization_view)

        # Set the stacked widget as the central widget
        self.grid_layout.addWidget(self.stacked_widget,0,1,1,1)

        # Create buttons to switch between views
        self.to_segmentation_btn = QRadioButton('Сегментация')
        self.to_segmentation_btn.toggled.connect(self.show_segmentation_view)
        self.to_segmentation_btn.setChecked(True)
        self.to_approximation_btn = QRadioButton('Аппроксимация')
        self.to_approximation_btn.toggled.connect(self.show_approximation_view)
        self.to_optimization_btn = QRadioButton('Оптимизация')
        self.to_optimization_btn.toggled.connect(self.show_optimization_view)

        # Add buttons to the main window
        switch_layout = QVBoxLayout()
        switch_layout.addWidget(self.to_segmentation_btn)
        switch_layout.addWidget(self.to_approximation_btn)
        switch_layout.addWidget(self.to_optimization_btn)
        switch_widget = QGroupBox()
        switch_widget.setLayout(switch_layout)
        self.grid_layout.addWidget(switch_widget,0,0,1,1)

        self.central_widget.setLayout(self.grid_layout)
        self.setCentralWidget(self.central_widget)

    def show_segmentation_view(self):
        self.stacked_widget.setCurrentIndex(0)

    def show_approximation_view(self):
        self.stacked_widget.setCurrentIndex(1)

    def show_optimization_view(self):
        self.stacked_widget.setCurrentIndex(2)

    def load_styles(self):
        style_file = "styles.css"
        with open(style_file, "r"):
            self.setStyleSheet(open(style_file).read())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    apply_stylesheet(app,theme='light_cyan_500.xml')
    main_window = MainApplication()
    main_window.show()
    sys.exit(app.exec_())