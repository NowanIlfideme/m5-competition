# M5 Modelling

## Setup

1. Create a conda environment called `m5-comp`:

    `conda env create`

2. Then, install the library in editable mode:

    `pip install -e .`

3. Set up your Kaggle API credentials, as outlined [here](https://github.com/Kaggle/kaggle-api#api-credentials).

4. Download and unzip the data:

    ```bash
    kaggle competitions download -c m5-forecasting-accuracy -p data

    unzip data/m5-forecasting-accuracy.zip -d data
    ```

5. Run jupyter.
