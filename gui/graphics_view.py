from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication, QFrame, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QMouseEvent, QWheelEvent, QPainter


class GraphicsView(QGraphicsView):
    def __init__(self, size: QSize, parent=None):
        super(GraphicsView, self).__init__(parent)
        self.setMinimumSize(size)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._pan = False
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.HighQualityAntialiasing)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setFrameStyle(QFrame.Shadow_Mask)
        self.setPlaceholderImage()

    def setPlaceholderImage(self):
        placeholderPixmap = QPixmap("placeholder.png")  # Replace with your placeholder image path
        scaledPixmap = placeholderPixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.set_image(placeholderPixmap)

    def set_image(self, pixmap):
        if not self._empty:
            self._scene.clear()
            self._zoom = 0
            self.resetTransform()
        self.pixmapItem = QGraphicsPixmapItem(pixmap)  # Add the pixmap as is
        self.scene().addItem(self.pixmapItem)
        self.scene().setSceneRect(self.pixmapItem.boundingRect())
        self.fitImageInView()
        self._empty = False

    def fitImageInView(self, preserveAspectRatio=True):
        if not self.scene() or self.scene().itemsBoundingRect().isEmpty():
            return

        rect = self.scene().itemsBoundingRect()
        self.fitInView(rect, Qt.KeepAspectRatio if preserveAspectRatio else Qt.IgnoreAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        zoomInFactor = 1.25  # Zoom-in factor
        zoomOutFactor = 1 / zoomInFactor

        # Save the scene pos
        oldPos = self.mapToScene(event.pos())

        # Zoom
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)

        # Get the new position
        newPos = self.mapToScene(event.pos())

        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self._pan = True
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
            self.setCursor(Qt.ClosedHandCursor)
        super(GraphicsView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)
        super(GraphicsView, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._pan:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - (event.x() - self._pan_start_x))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - (event.y() - self._pan_start_y))
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
        super(GraphicsView, self).mouseMoveEvent(event)
