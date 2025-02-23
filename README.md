# MedVKAN

MedVKAN: Efficient Feature Extraction with Mamba and KAN for Medical Image Segmentation

## Environment Install

```shell
conda create -n medvkan python=3.10
conda activate medvkan

pip install torch==2.0.1 torchvision==0.15.2
pip install causal-conv1d==1.1.1
pip install mamba-ssm==2.2.2
pip install torchinfo timm numba
```
```shell
git clone https://github.com/beginner-cjh/MedVKAN
cd MedVKAN/medvkan
pip install -e .
```

## Datasets

You can download the AbdomenMRI / Microscopy / BUSI / ACDC / COVID-19 dataset from the [link](https://drive.google.com/drive/folders/1CH2OWQpd4Sa-BES6oFLRC469gTxf6QUO?usp=drive_link)

Place in the data folder (../data/nnUNet_raw) . 

The data structure will be in this format：

```shell
../data/nnUNet_raw/Dataset703_NeurlPSCell
├── imagesTr
│   ├── cell_00001_0000.png
│   ├── ...
├── imagesTs
│   ├── cell_00001_0000.png
│   ├── ...
├── labelsTr
│   ├── cell_00001.png
│   ├── ...
├── labelsTs
│   ├── cell_00001_label.tiff
│   ├── ...
├── dataset.json
```

Then pre-process the dataset with the following command :

```shell
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

```shell
##such as Microscopy
nnUNetv2_plan_and_preprocess -d 703 --verify_dataset_integrity
```

Then you need to change the batch size in the file：

```shell
../data/nnUNet_raw/Dataset703_NeurlPSCell/nnUNetPlans.json
"batch_size": 8
```

Specifically, for the ACDC dataset you also need to change the patch size（This is the size of the input image）：

```shell
../data/nnUNet_raw/Dataset027_ACDC/nnUNetPlans.json
"patch_size": [
    256,
    256
],
```

## Training & Evaluation

Using the following command to train & evaluate MedVKAN

```shell
bash scripts/train_{Datasets}.sh nnUNetTrainerMedVKAN
```
Datasets can be AbdomenMR / BUSI / Microscopy / ACDC / COVID , such as:
```shell
#Microscopy Dataset
bash scripts/train_Microscopy.sh nnUNetTrainerMedVKAN
```

You can download our model checkpoints [here](https://drive.google.com/drive/folders/1Krgcbz31IA2QfiXtjRKALvl40XMjejXc?usp=drive_link).

## Acknowledgements

We thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba), [UMamba](https://github.com/bowang-lab/U-Mamba), [VMamba](https://github.com/MzeroMiko/VMamba), [UKAN](https://github.com/CUHK-AIM-Group/U-KAN) and [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) for making their valuable code & data publicly available.

## Citation

```
@article{MedVKAN,
    title={MedVKAN: Efficient Feature Extraction with Mamba and KAN for Medical Image Segmentation},
    author={},
    journal={},
    year={2025}
}
```
