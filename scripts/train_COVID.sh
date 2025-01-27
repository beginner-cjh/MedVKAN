MAMBA_MODEL=$1
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset001_COVID-19/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
TRUE_IMAGE_PATH="data/nnUNet_raw/Dataset001_COVID-19/imagesTs"
TRUE_LABEL_PATH="data/nnUNet_raw/Dataset001_COVID-19/labelsTs"
GPU_ID="0"

#Train
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 001 2d all -tr ${MAMBA_MODEL} -num_gpus 1 &&
##If you use our trained model, comment out this line of training code

#Predict
echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "${TRUE_IMAGE_PATH}" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 001 \
    -c 2d \
    -tr "${MAMBA_MODEL}" \
    --disable_tta \
    -f all \
    -chk "checkpoint_best.pth" &&
    ##If you are using our trained model, change checkpoint_best.pth to COVID19.pth

#Compute Metrics
echo "Computing iou and F1..."
python evaluation/COVID_metrics.py \
    -t "${TRUE_LABEL_PATH}" \
    -p "${PRED_OUTPUT_PATH}"&&
    
echo "Done."
