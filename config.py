from torchvision.transforms import transforms

from network.helper import UnNormalize


class BaseConfig:
    Evaluation = False
    VIDEO = False
    # train conf
    EPOCH = 100
    TRAIN = 'train'
    TEST = 'test'
    # data config
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    NUM_CLASSES = 2
    NUM_FRAMES = 4
    BATCH_SIZE = 20
    FRAMES_STEP = 1
    base_lr = 1e-4
    device_ids = [0, 1]
    shuffle = True
    IS_DISTRIBUTION = False
    denormalize = None
    checkpoint = 'checkpoint'
    type = ''
    ALL_DIM = 256

    mean = [0.485, 0.456, 0.406]  # RGB
    std = [0.229, 0.224, 0.225]  # RGB
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    def __init__(self, mode, set_path, rank):
        self.mode = mode
        self.set_path = set_path
        self.rank = rank


class PVT2Config(BaseConfig):
    PATCH_SIZE = 7
    IMAGE_SIZE = 224
    IN_CHANNELS = 12
    image_based = True
    BATCH_SIZE = 64
    NUM_FRAMES = 1
    FRAMES_STEP = NUM_FRAMES
    base_lr = 1e-3

    # model config
    EMBED_DIMS = [64, 128, 320, 512]
    NUM_HEADS = [1, 2, 5, 8]
    MLP_RATIOS = [8, 8, 4, 4]
    QKV_BIAS = True
    QK_SCALE = None
    DROP_RATE = 0.1
    ATTN_DROP_RATE = 0.0
    DROP_PATH_RATE = 0.1
    DEPTHS = [3, 4, 8, 3]
    SR_RATIOS = [8, 4, 2, 1]
    NUM_STAGES = 4
    LINEAR = False
    NUM_CLASSES = 1

    mean = [0.447, 0.450, 0.417]
    std = [0.220, 0.220, 0.220]
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


class HRTSmallConfig(BaseConfig):
    DROP_PATH_RATE = 0.2
    NUM_CLASSES = 1
    in_channels = 480
    hidden_dim = 512

    class STAGE1:
        NUM_MODULES = 1
        NUM_BRANCHES = 1
        NUM_BLOCKS = [2]
        NUM_CHANNELS = [64]
        NUM_HEADS: [2]
        NUM_MLP_RATIOS: [4]
        NUM_RESOLUTIONS: [[56, 56]]
        BLOCK = "BOTTLENECK"

    class STAGE2:
        NUM_MODULES = 1
        NUM_BRANCHES = 2
        NUM_BLOCKS = [2, 2]
        NUM_CHANNELS = [32, 64]
        NUM_HEADS = [1, 2]
        NUM_MLP_RATIOS = [4, 4]
        NUM_RESOLUTIONS = [[56, 56], [28, 28]]
        NUM_WINDOW_SIZES = [7, 7]
        BLOCK = "POOL_FORMER_BLOCK"

    class STAGE3:
        NUM_MODULES = 4
        NUM_BRANCHES = 3
        NUM_BLOCKS = [2, 2, 2]
        NUM_CHANNELS = [32, 64, 128]
        NUM_HEADS = [1, 2, 4]
        NUM_MLP_RATIOS = [4, 4, 4]
        NUM_RESOLUTIONS = [[56, 56], [28, 28], [14, 14]]
        NUM_WINDOW_SIZES = [7, 7, 7]
        BLOCK = "POOL_FORMER_BLOCK"

    class STAGE4:
        NUM_MODULES = 2
        NUM_BRANCHES = 4
        NUM_BLOCKS = [2, 2, 2, 2]
        NUM_CHANNELS = [32, 64, 128, 256]
        NUM_HEADS = [1, 2, 4, 8]
        NUM_MLP_RATIOS = [4, 4, 4, 4]
        NUM_RESOLUTIONS = [[56, 56], [28, 28], [14, 14], [7, 7]]
        NUM_WINDOW_SIZES = [7, 7, 7, 7]
        BLOCK = "TRANSFORMER_BLOCK"


class MaskConfig(HRTSmallConfig):
    image_based = False
    NUM_FRAMES = 4
    BATCH_SIZE = 20
    FRAMES_STEP = 4
    IS_DISTRIBUTION = False
    NUM_CLASSES = 1
    shuffle = True
    base_lr = 1e-4
    mean = [0.480, 0.480, 0.450]
    std = [0.203, 0.178, 0.193]
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    denormalize = UnNormalize(mean=mean, std=std)


class SplicingMaskConfig(MaskConfig):
    type = 'Splicing'
    NUM_FRAMES = 4
    BATCH_SIZE = 10
    FRAMES_STEP = 4
    base_lr = 1e-4
    mean = [0.447, 0.450, 0.417]
    std = [0.220, 0.220, 0.220]
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


class ImageSplicingMaskConfig(SplicingMaskConfig):
    type = 'ImageSplicing'
    BATCH_SIZE = 32
    FRAMES_STEP = 1
    NUM_FRAMES = 1
    base_lr = 1e-3
    image_based = True


class ClsMaskConfig(SplicingMaskConfig):
    type = 'ClsMask'
    NUM_FRAMES = 1
    BATCH_SIZE = 64
    FRAMES_STEP = NUM_FRAMES
    NUM_CLASSES = 2
    classify = True
    base_lr = 1e-4
    NUM_CLASSES = 1
    image_based = True


class DFSConfig(HRTSmallConfig):
    NUM_CLASSES = 5
    shuffle = True
    NUM_FRAMES = 1
    image_based = True
    BATCH_SIZE = 36
    IS_DISTRIBUTION = False
    FRAMES_STEP = 1
    base_lr = 1e-5
    mean = [0.524, 0.434, 0.391]
    std = [0.229, 0.210, 0.193]
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    denormalize = UnNormalize(mean=mean, std=std)
