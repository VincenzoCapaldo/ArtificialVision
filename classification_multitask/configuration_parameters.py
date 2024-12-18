# configuration parameters for dataset configuration, used for dataset preprocessing and training and validation split
class DatasetConfig:
    IMAGE_SIZE: tuple = (224, 224)  # width, height. It's commonly used.
    CHANNELS: int = 3  # image channels (RGB = 3)
    NUM_CLASSES: int = 3  # gender, bag, hat
    VALIDATION_DIM: float = 0.2  # divide training and validation in 80% and 20%

    # resnet50 lavora su input normalizzati e il modello è stato allenato sul dataset ImageNet che ha i seguenti valori
    # di media e deviazione standard (ma non sono sicuro se dobbiamo usare questi o dobbiamo calcolarli sul nostro
    # dataset (o se ne abbiamo bisogno) visto che all'esame le condizioni saranno praticamente simili: stesso POV,
    # stesso ambiente ecc).
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD: tuple = (0.229, 0.224, 0.225)
    # path for train resources
    TRAIN_IMG_PATH: str = "../../Dataset/training_set/training_set/"
    TRAIN_LABEL_PATH: str = "../../Dataset/training_set.txt"
    # path for test resources
    TEST_IMG_PATH: str = "../../Dataset/validation_set/validation_set/"
    TEST_LABEL_PATH: str = "../../Dataset/validation_set.txt"


# configuration parameters for training phase
class TrainingConfig:
    BATCH_SIZE: int = 64
    NUM_EPICHS: int = 100
    INIT_LR: float = 1e-4
    OPTIMIZER_NAME: str = "Adam"
    BACKBONE_NAME: str = "resnet50"
