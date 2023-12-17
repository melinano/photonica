from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QFileDialog, QTableWidget, QLineEdit, QItemDelegate, QPushButton, QGridLayout, QWidget, \
    QTableWidgetItem
from PyQt5.QtGui import QDoubleValidator
from gui.graphics_view import GraphicsView

class NumericDelegate(QItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        validator = QDoubleValidator(editor)
        editor.setValidator(validator)
        return editor

class ApproximationView(QWidget):
    def __init__(self):
        super().__init__()
        # self.shared_data = shared_data
        self.setupUI()

    def setupUI(self):
        self.coeffInputParams = {
            "Коэфф2": 8,
            "Коэфф4": 5
        }

        # Grid layout
        self.view_layout = QGridLayout()

        self.loadCoordsBtn = QPushButton("Загрузить Координаты")
        self.processBtn = QPushButton("Обработать")

        self.loadCoordsBtn.clicked.connect(self.loadCoords)
        self.processBtn.clicked.connect(self.process)


        self.view_layout.addWidget(self.loadCoordsBtn, 0, 0, 1, 1)  # Row 0, Column 0
        self.view_layout.addWidget(self.processBtn, 0, 1, 1, 1)  # Row 0, Column 1
        self.setupInputTable(self.coeffInputParams)
        self.setupOutputGraphicView()
        self.setLayout(self.view_layout)

    def setupOutputGraphicView(self):
        self.outputGraphicsView = GraphicsView(QSize(600,400),self)
        self.view_layout.addWidget(self.outputGraphicsView, 1, 0,-1,1)  # Row 0, Column 0

    def setupInputTable(self, inputParamNames):
        self.inputTableWidget = QTableWidget()
        self.inputTableWidget.setRowCount(len(inputParamNames))  # Set the number of parameters
        self.inputTableWidget.setColumnCount(2)
        self.inputTableWidget.setMinimumSize(QSize(120, 100))

        self.inputTableWidget.setHorizontalHeaderLabels(['Коэффицент', 'Значение'])

        numericDelegate = NumericDelegate(self.inputTableWidget)
        self.inputTableWidget.setItemDelegateForColumn(1, numericDelegate)

        self.inputTableWidget.horizontalHeader().setStretchLastSection(True)
        self.inputTableWidget.verticalHeader().setVisible(False)  # Hide the vertical header

        self.initInputTable(inputParamNames)

        self.view_layout.addWidget(self.inputTableWidget, 1, 1, -1, 1)

    def initInputTable(self, inputParams):
        for i, name in enumerate(inputParams):
            self.inputTableWidget.setItem(i, 0, self.getNameTableItem(name))
            self.inputTableWidget.setItem(i, 1, self.getValueTableItem(str(inputParams[name])))

        # Resize the first column to fit content
        self.inputTableWidget.resizeColumnToContents(0)

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

    def loadCoords(self):
        # Open a dialog to select an image file
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Textfile", "",
                                                  "Text Files (*.txt)")

        if filePath:
            # DO SOMETHING

            pass


    def process(self):
        # Logic to process the image,

        pass