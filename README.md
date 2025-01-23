# MedVKAN

 [MedVKAN: An Efficient Feature Extraction Method Integrating Mamba and KAN for Enhanced Medical Image Segmentation]()*

<img src="https://github.com/beginner-cjh/MedVKAN/blob/main/assets/MedVKAN.png" width="80%" />

## Main Results

- Microscopy_BUSI_AbdomenMRI
<img src="https://github.com/beginner-cjh/MedVKAN/blob/main/assets/Results_Microscopy_BUSI_AbdomenMRI.PNG" width="70%" />

- ACDC_COVID19
<img src="https://github.com/beginner-cjh/MedVKAN/blob/main/assets/Results_ACDC_COVID19.PNG" width="70%" />

## Contact
If you have any questions about our project, please do not hesitate to contact us by sending an e-mail to cjh1282980418@163.com.

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

You can download the AbdomenMRI and Microscopy dataset from [U-Mamba](https://github.com/bowang-lab/U-Mamba) .

You can download the BUSI / ACDC / COVID-19 dataset from the [link](https://drive.google.com/drive/folders/1CH2OWQpd4Sa-BES6oFLRC469gTxf6QUO?usp=drive_link)

Place in the data folder (../data/nnUNet_raw) . 

Then pre-process the dataset with the following command :

```shell
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
##such as Microscopy
nnUNetv2_plan_and_preprocess -d 703 --verify_dataset_integrity
```


## Training & Evaluation

Using the following command to train & evaluate MedVKAN

```shell
bash scripts/train_{Datasets}.sh nnUNetTrainerMedVKAN
```
Datasets can be AbdomenMRI / BUSI / Microscopy / ACDC / COVID , such as:
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
    title={MedVKAN: An Efficient Feature Extraction Method Integrating Mamba and KAN for Enhanced Medical Image Segmentation},
    author={},
    journal={arXiv preprint arXiv:},
    year={2025}
}
```
