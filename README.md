# Fake-News-Detection
My capstone project for BSDA degree programme at Sunway University:

<u>**Fake News Detection Using Social Media User Network and Engagement Features**</u>
***
For this project, `./code/config.json` is configured to download the required files only. Credit of data retrieval scripts and dataset goes to [KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).

(Please refer to `README.md` in [KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) for data retrieval instructions.)

Then, `data_extract.py` is used to extract and clean the data to be used for this project. The output is `final_dataset.csv`.

The config file `config.yaml` defines the best model chosen after experimentation and model selection (in this case, Random Forest with 80% train size), and will be used to train the final model by running `model.py`.

To train the model from the beginning, which includes the step of model selection, remove the values of `algorithm` and `train_size` from the `config.yaml` file, then run `model.py`.

Finally, the model will be output as `model.pkl`.
