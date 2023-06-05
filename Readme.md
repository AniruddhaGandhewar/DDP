Code
- VIS-NIR Apple Spectroscopy
    - Data Preprocessing - Apples.ipynb: Averages intensity measurements and converts them to absorbance values.

    - models.py: Contain code for the following models -
        - PLS_DA
        - SPCA_LR
        - PCR
        - SFS_RR (referred as SFS_LR in code)
        - QT_SFS_RR (referred as SFS_QR in code)
        - PLSR

    - Exploratory Data Analysis - All Apples.ipynb: EDA excercise for spectra of all apples.

    - Modelling - All Apples.ipynb: Experiments on all apples' spectra using the following models  -
        - PLSR
        - PCR
        - SPCA_LR

    - Modelling 2 - All Apples.ipynb: Experiments on all apples' spectra using the following models -
        - SFS_RR
        - QT_SFS_RR
        - SVR
        - VR

    - Exploratory Data Analysis - Turkish.ipynb: EDA excercise for spectra of Turkish apples.

    - Modelling - Turkish.ipynb: Experiments on Turkish apples' spectra using the following models  -
        - PLSR
        - PCR
        - SPCA_LR

    - Modelling 2 - Turkish.ipynb: Experiments on Turkish apples' spectra using the following models -
        - SFS_RR

    - Classification Polish vs. Turkish.ipynb: PLS-DA classification model for classifying Polish and Turkish apples.


- NIR Honey Spectroscopy
    - Honey - PLS-DA.ipynb: PLS-DA applied to NIR Honey Spectra (and it's lower resolution variants).

    - Honey - SNN.ipynb: Siamese Dataset, Siamese Neural Network and a training function implemented in PyTorch.

Data
- Apple VIS-NIR Spectroscopic Dataset
    - AvgAbs.csv: Spectra of all apples post sample averaging.

    - NoisyAbs.csv: Spectra of all apples.

    - PolishAbs.csv: Spectra of Polish apples.

    - TurkishAbs.csv: Spectra of Turkish apples.

    - GridSearch.xlsx: Grid search results of QT_SFS_RR.
- Honey NIR Spectroscopic Dataset
    - AugmentedNIRS.csv: Augmented (additional spectra generated) NIR Spectroscopic dataset of Honey samples.

    - honey_nirs.csv: Original NIR Spectroscopic dataset dataset of Honey samples.

    - honey_nirs.rdata: Original NIR Spectroscopic dataset dataset of Honey samples in .rdata format.

    - nirs_train.csv: Train split.

    - nirs_val.csv: Validation split.