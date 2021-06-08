# Movie Recommendation System

This model achieved RSME: 0.8205079606996796 after 5 epochs.

## Requirements
```
#run the following command to install required packages
pip install numpy pandas==1.1.5 scipy

# install torch according to the system's hardware configuration
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

```python
# to train the network
python train.py

#to predict movies
python predict.py #default user Id is set to 100

#to predict movies for any user Id 
python predict.py -u <User Id>