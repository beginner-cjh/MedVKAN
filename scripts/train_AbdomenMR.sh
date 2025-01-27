MAMBA_MODEL=$1
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset702_AbdomenMR/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
EVAL_METRIC_PATH="data/nnUNet_results/Dataset702_AbdomenMR/${MAMBA_MODEL}__nnUNetPlans__2d"
GPU_ID="1"

#Train
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 702 2d all -tr ${MAMBA_MODEL} -num_gpus 1 &&
##If you use our trained model, comment out this line of training code

#Predict
echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "data/nnUNet_raw/Dataset702_AbdomenMR/imagesTs" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 702 \
    -c 2d \
    -tr "${MAMBA_MODEL}" \
    --disable_tta \
    -f all \
    -chk "checkpoint_best.pth" &&
    ##If you are using our trained model, change checkpoint_best.pth to AbdomenMRI.pth

#Compute Metrics
echo "Computing dice..."
python evaluation/abdomen_DSC_Eval.py \
    --gt_path "data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_DSC.csv"  &&

echo "Computing NSD..."
python evaluation/abdomen_NSD_Eval.py \
    --gt_path "data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_NSD.csv" &&

echo "Done."
