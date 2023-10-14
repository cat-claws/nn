import torchvision

def fasterrcnn_mobilenet_v3_large_320_fpn(weights = 'DEFAULT', num_classes = 2):
	model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights = weights)
	model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes = num_classes)
	return model
