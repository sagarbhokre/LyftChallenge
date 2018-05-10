echo $1

TRN_IMG_PATH=./data/Train/CameraRGB/
TRN_ANN_PATH=./data/Train/CameraSeg/
TST_IMG_PATH=./data/Train/CameraRGB/
TST_ANN_PATH=./data/Train/CameraSeg/
N_CLASS=12
WIDTH=800
HEIGHT=600
EPOCHS=40

EPOCH=$2

if [ $1 == "train" ] ; then
    echo "python  train.py  --save_weights_path=weights/ex1  --train_images=\"${TRN_IMG_PATH}\"  --train_annotations=\"${TRN_ANN_PATH}\"  --val_images=\"${TST_IMG_PATH}\"  --val_annotations=\"${TST_ANN_PATH}\"  --n_classes=$N_CLASS  --input_height=${HEIGHT}  --input_width=${WIDTH}  --model_name=\"vgg_segnet\" --epochs $EPOCHS --optimizer_name=\"adadelta\"" > exec_cmd
elif [ $1 == "predict" ] ; then
    export DISPLAY=:0
    echo "python  predict.py  --save_weights_path=weights/ex1  --test_images=\"${TST_IMG_PATH}\"  --n_classes=$N_CLASS  --input_height=${HEIGHT}  --input_width=${WIDTH}  --model_name=\"vgg_segnet\" --epoch_number=${EPOCH}" > exec_cmd
elif [ $1 == "visualize" ] ; then
    export DISPLAY=:0
    echo "python  visualizeDataset.py  --images=\"${TRN_IMG_PATH}\"  --annotations=\"${TRN_ANN_PATH}\"  --n_classes=$N_CLASS" > exec_cmd
elif [ $1 == "demo" ] ; then
    export DISPLAY=:0
    echo "python demo.py  Example/test_video.mp4" > exec_cmd
else
    echo "Invalid args"
fi
. exec_cmd
rm exec_cmd
exit
