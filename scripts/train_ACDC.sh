MAMBA_MODEL=$1
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset027_ACDC/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
TRUE_IMAGE_PATH="data/nnUNet_raw/Dataset027_ACDC/imagesTs"
TRUE_LABEL_PATH="data/nnUNet_raw/Dataset027_ACDC/labelsTs"
GPU_ID="1"

CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 027 2d all -tr ${MAMBA_MODEL} -num_gpus 1 &&

echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "${TRUE_IMAGE_PATH}" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 027 \
    -c 2d \
    -tr "${MAMBA_MODEL}" \
    --disable_tta \
    -f all \
    -chk "checkpoint_best.pth" &&

echo "Computing DSC and NSD..."
python evaluation/ACDC_DSC.py \
    --gt_path "${TRUE_LABEL_PATH}" \
    --seg_path "${PRED_OUTPUT_PATH}" &&

python evaluation/ACDC_NSD.py \
    --gt_path "${TRUE_LABEL_PATH}" \
    --seg_path "${PRED_OUTPUT_PATH}" &&
echo "Done."
