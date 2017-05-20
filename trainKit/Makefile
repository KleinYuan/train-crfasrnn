Train-single-gpu-from-0:
	${CAFFE_PATH}/caffe/build/tools/caffe train -solver ${TRAIN_CRF_RNN_PATH}/CRFRNN_solver.prototxt -weights fcn-8s-pascal.caffemodel -gpu 0

Train-multiple-gpus-from-0:
	${CAFFE_PATH}/caffe/build/tools/caffe train -solver ${TRAIN_CRF_RNN_PATH}/CRFRNN_solver.prototxt -weights fcn-8s-pascal.caffemodel -gpu 0,1

Train-single-gpu-fine-tuning:
	${CAFFE_PATH}/caffe/build/tools/caffe train -solver ${TRAIN_CRF_RNN_PATH}/CRFRNN_solver.prototxt -weights TVG_CRFRNN_COCO_VOC.caffemodel  -gpu 0

Train-multiple-gpus-fine-tuning:
	${CAFFE_PATH}/caffe/build/tools/caffe train -solver ${TRAIN_CRF_RNN_PATH}/CRFRNN_solver.prototxt -weights TVG_CRFRNN_COCO_VOC.caffemodel -gpu 0,1