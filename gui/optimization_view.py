from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QGridLayout, QWidget, QItemDelegate, QLineEdit, QGroupBox, \
    QVBoxLayout, QRadioButton, QTableWidget, QTableWidgetItem, QGraphicsWidget
from PyQt5.QtGui import QDoubleValidator, QFont

from gui.graphics_view import GraphicsView


# Matplotlib canvas class to create figure
class Canvas(FigureCanvas):
    def __init__(self, parent=QGraphicsWidget, width=5, height = 4, dpi = 200):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(Canvas, self).__init__(self.fig)

class NumericDelegate(QItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        validator = QDoubleValidator(editor)
        editor.setValidator(validator)
        return editor

class OptimizationView(QWidget):
    def __init__(self):
        super().__init__()
        #self.shared_data = shared_data
        self.setupUI()

    def setupUI(self):
        self.morphInputParams = {
            "Глубина": 0,
            "Период": 0,
            "Коэфф. заполнения": 0,
            "Диам. модификации": 0,
        }

        self.coeffInputParams = {
            "Коэфф1":0,
            "Коэфф2":8,
            "Коэфф3":1,
            "Коэфф4":5
        }

        self.outputParams = {
            "λ (nm)": 0,
            "Regime": 0,
            "Pulse Picker": 0,
            "Frequency (kHz)": 0,
            "Power (mW)": 0,
            "Burst" : 0,
            "Compressor" : 0,
            "v/vi" :0 ,
            "dy (μm)": 0,
            "dx (μm)" : 0,
            "dz (μm)" : 0,
            "Transmittance (%)" : 0,
            "TruePower (mW)" : 0
        }

        self.view_layout = QGridLayout()


        # Setup different UI components
        self.setupPlotGraphicView()
        self.setupStartButton()
        self.setupRadioGroupBox()
        self.setupRadioButtons()

        if self.morphologyRadioButton.isChecked():
            inputParams = self.morphInputParams
        else: inputParams = self.coeffInputParams

        self.setupInputTable(inputParams)
        self.setupOutputHeader()
        self.setupOutputTable(self.outputParams)
        self.setLayout(self.view_layout)

        #self.coefficientRadioButton.toggled.connect(self.updateInputForm)

    def setupPlotGraphicView(self):
        self.plotWidget = Canvas(width=3, height=2, dpi=100)
        self.plotWidget.setMinimumSize(QSize(600, 400))
        self.plotWidget.setObjectName("plotWidget")
        self.view_layout.addWidget(self.plotWidget, 0, 0, -1, 2)
        self.plotWidget.ax.plot(range(0, 100), range(0, 200, 2))

    def setupRadioGroupBox(self):
        self.radioGroupBox = QGroupBox()
        self.radioGroupBox.setMinimumSize(QSize(200, 40))
        self.radioGroupBox.setTitle("Параметры ввода")
        self.radioGroupBox.setObjectName("radioGroupBox")
        self.view_layout.addWidget(self.radioGroupBox, 0, 2, 2, 1)
        self.radioBoxLayout = QVBoxLayout()
        self.radioGroupBox.setLayout(self.radioBoxLayout)


    def setupRadioButtons(self):
        self.radioBoxLayout.setObjectName("radioButtonsLayout")

        self.coefficientRadioButton = QRadioButton(self.radioGroupBox)
        self.coefficientRadioButton.setText("Коэффициенты формулы")
        self.coefficientRadioButton.setObjectName("coefficientRadioButton")
        self.radioBoxLayout.addWidget(self.coefficientRadioButton)

        self.morphologyRadioButton = QRadioButton(self.radioGroupBox)
        self.morphologyRadioButton.setText("Параметры морфологии")
        self.morphologyRadioButton.setChecked(True)
        self.morphologyRadioButton.setObjectName("morphologyRadioButton")
        self.radioBoxLayout.addWidget(self.morphologyRadioButton)

        self.coefficientRadioButton.toggled.connect(self.updateInputTable)
        self.morphologyRadioButton.toggled.connect(self.updateInputTable)

    def setupInputTable(self, inputParamNames):
        self.inputTableWidget = QTableWidget()
        self.inputTableWidget.setRowCount(len(inputParamNames))  # Set the number of parameters
        self.inputTableWidget.setColumnCount(2)
        self.inputTableWidget.setMinimumSize(QSize(120, 100))

        self.inputTableWidget.setHorizontalHeaderLabels(['Parameter', 'Value'])

        numericDelegate = NumericDelegate(self.inputTableWidget)
        self.inputTableWidget.setItemDelegateForColumn(1, numericDelegate)

        self.inputTableWidget.horizontalHeader().setStretchLastSection(True)
        self.inputTableWidget.verticalHeader().setVisible(False)  # Hide the vertical header

        self.initInputTable(inputParamNames)

        self.view_layout.addWidget(self.inputTableWidget, 2, 2, 1, 1)

    def initInputTable(self, inputParams):
        for i, name in enumerate(inputParams):
            self.inputTableWidget.setItem(i, 0, self.getNameTableItem(name))
            self.inputTableWidget.setItem(i, 1, self.getValueTableItem(str(inputParams[name])))

        # Resize the first column to fit content
        self.inputTableWidget.resizeColumnToContents(0)

    def updateInputTable(self):
        # Check which radio button is checked and set parameters accordingly
        if self.morphologyRadioButton.isChecked():
            parameters = self.morphInputParams
        else:
            parameters = self.coeffInputParams

        # Update the table with new parameters
        self.populateInputTable(parameters)

    def populateInputTable(self, parameters):
        self.inputTableWidget.setRowCount(len(parameters))
        for i, name in enumerate(parameters):
            self.inputTableWidget.setItem(i, 0, self.getNameTableItem(name))
            self.inputTableWidget.setItem(i, 1, self.getValueTableItem(str(parameters[name])))

    def getNameTableItem(self, name):
        # Non-editable item for parameter name
        nameItem = QTableWidgetItem(name)
        nameItem.setFlags(Qt.ItemIsEnabled)  # Makes the cell non-editable
        nameItem.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

        return nameItem

    def getValueTableItem(self, value):
        # Non-editable item for parameter name
        valueItem = QTableWidgetItem(value)
        valueItem.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

        return valueItem

    def setupStartButton(self):
        self.startButton = QPushButton()
        self.startButton.setText("Пуск!")
        self.startButton.setObjectName("startButton")
        self.view_layout.addWidget(self.startButton, 3, 2, 2, 1)
        self.startButton.clicked.connect(self.updateOutputTable)

    def setupOutputHeader(self):
        font = QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(11)

        # Header of Output Form
        self.resultLabel = QLabel()
        self.resultLabel.setFont(font)
        self.resultLabel.setObjectName("resultLabel")
        self.view_layout.addWidget(self.resultLabel, 0, 4, 1, 1)

    def setupOutputTable(self, outputParams):
        self.outputTableWidget = QTableWidget()
        self.outputTableWidget.setMinimumSize(QSize(200, 300))
        self.outputTableWidget.setRowCount(len(self.outputParams))  # Set the number of parameters
        self.outputTableWidget.setColumnCount(2)
        self.view_layout.addWidget(self.outputTableWidget, 0, 4, -1, 1)

        self.outputTableWidget.setHorizontalHeaderLabels(['Parameter', 'Value'])

        self.outputTableWidget.horizontalHeader().setStretchLastSection(True)
        self.outputTableWidget.verticalHeader().setVisible(False)  # Hide the vertical header

        self.populateOutputTable(outputParams)

    def populateOutputTable(self, outputParams):
        for i, name in enumerate(self.outputParams):
            self.outputTableWidget.setItem(i, 0, self.getNameTableItem(name))

            if not isinstance(outputParams[name],str):
                valueItem = self.getValueTableItem(str(outputParams[name]))
            else: valueItem = self.getValueTableItem(str(outputParams[name]))
            valueItem.setFlags(Qt.ItemIsEnabled) # Makes the cell non-editable
            self.outputTableWidget.setItem(i, 1, valueItem)

        # Resize the first column to fit content
        self.outputTableWidget.resizeColumnToContents(0)

    def updateOutputTable(self):
        outputParams = {
            "λ (nm)": 42,
            "Regime": 42,
            "Pulse Picker": 42,
            "Frequency (kHz)": 42,
            "Power (mW)": 42,
            "Burst": 42,
            "Compressor": 42,
            "v/vi": 42,
            "dy (μm)": 24,
            "dx (μm)":"42",
            "dz (μm)": "42",
            "Transmittance (%)": 42,
            "TruePower (mW)": 42
        }
        self.populateOutputTable(outputParams)

