# Backbones and baseline models

## SimSwap
1. Create a dedicated conda environment
```bash
conda create -n simswap python=3.8
conda activate simswap
```
2. Install antelope:
```bash
mkdir insightface_func/models
pip install gdown
gdown 1goH5lO8BAhTpRhpBeXqWEcGkxiiLlgx9
unzip antelope.zip -d insightface_func/models
```
3. Install checkpoints:
```bash
cd ../../
mkdir arcface_model
gdown 1TLNdIufzwesDbyr_nVTR7Zrx9oRHLM_N
mv arcface_checkpoint.zip arcface_model/
gdown 1PXkRiBUYbu1xWpQyDEJvGKeqqUFthJcI
unzip checkpoints.zip -d checkpoints
```

4. Install dependencies:
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 https://download.pytorch.org/whl/torch_stable.html
pip install --ignore-installed imageio
pip install insightface==0.2.1 onnxruntime moviepy
```

5. Run face swap:
```bash
python test_one_image.py --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path source_image.jpg --pic_b_path target_image.jpg --output_path output/ --use_encoded_image

python test_video_swapsingle.py --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path source_image.jpg --video_path target_video.mp4 --output_path ./output/output_video.mp4 --name people
```
