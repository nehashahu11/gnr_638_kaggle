# Generating High Resolution Zoom-IN for Images using LMLT-Base-x2 (Low-to-high Multi-Level Vision Transformer)

Russel Abreo ,Anand Patel


### Requirements
```
# Install Packages
pip install -r requirements.txt
pip install matplotlib

# Install BasicSR
python3 setup.py develop
```


### Dataset
The model is trained on DIV2K, Flickr2K.
You can download two datasets at https://github.com/dslisleedh/Download_df2k/blob/main/download_df2k.sh
and prepare other test datasets at https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#Common-Image-SR-Datasets.

The model is finetuned on RealSR dataset. The link for the dataset is https://www.kaggle.com/datasets/yashchoudhary/realsr-v1.

### Preprocessing the dataset

Preprocessing code can be found in process_real_sr_dataset.py and extract_subimages_realsr.py.

Creating a compatible directory structure.
```
python process_real_sr_dataset.py --directory_path "path_to_dataset_directory" --new_directory_name "realsr"  --scale "2"
```
And also, you'd better extract subimages using 
```
python3 scripts/data_preparation/extract_subimages_realsr.py
```

By running the code above, you will get subimages of RealSR dataset.


### Finetune
The configuration for finetuning is present in options/finetune/LMLT/finetune_base_RealSR_X2.yml .
You can finetune LMLT with the following command below.
```
python3 basicsr/train.py -opt options/finetune/LMLT/finetune_base_RealSR_X2.yml
```

Finetuning code can be found in basicsr/models/sr_model.py where the Knowledge Distillation approach along with Knowledge Distillation Loss is implemented.


### Test
You can test LMLT following commands below
```
python3 basicsr/test.py -opt options/test/LMLT/test_base_benchmark_X2.yml
```

### Novelty 
Novelty code can be found in the branch "novelty-task". Added lpips loss in basicsr/metrics/psnr_ssim.py. The UI changes are added in streamlit-app.py , preprocessing and postprocessing code is added in sr_model for finetuning purpose. Model is finetuned on unplash2k dataset https://github.com/dongheehand/unsplash2K

### NOTE
- Refer the lmlt_notebook.ipynb for finetuning steps.
Refer the IE643_Final_Streamlit_Interface_Crop.ipynb for the Interface.
The interface code is mentioned in streamlit.py.
- To run the interface using streamlit , refer the IE643_Final_Streamlit_Interface_Crop.ipynb notebook and consider the following repository https://github.com/abreorussel/super-resolution-streamlit-app
- The fintuned and the pretrained model can be found in experiments/models and experiments/pretrained_model respectively.
- experiments.py contains all the experiments performed.


## Credits

This project builds upon the work of others:

- **Research Paper**:  
  [*LMLT: Low-to-high Multi-Level Vision Transformer for Image Super-Resolution*](https://www.arxiv.org/abs/2409.03516) by Jeongsoo Kim, Jongho Nang, Junsuk Choe<sup>*</sup>. Published in 2024*.

- **Code Repository**:  
  Original implementation by [jwgdmkj](https://github.com/jwgdmkj/LMLT/tree/main) on GitHub.

