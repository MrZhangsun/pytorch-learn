from ultralytics import YOLO
import ultralytics
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
# Load a pretrained YOLO model
model = YOLO("yolo26n.pt")

log.info(f"Model loaded, version: {ultralytics.__version__}, yolo version: {model._version}", )
# Perform object detection on an image
results = model("https://ultralytics.com/images/bus.jpg")

# Visualize the results
for result in results:
    result.show()