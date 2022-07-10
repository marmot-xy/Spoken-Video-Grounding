import math

from yacs.config import CfgNode as CN

_c = CN()
_c.model_dim = 1024
_c.max_word_num = 20
_c.max_frame_num = 64
_c.max_audio_num = 800
_c.audio_dim = 128
_c.chunk_num = 8
_c.core_dim = _c.model_dim // _c.chunk_num


# Model Config
_c.model = CN()
_c.model.name = "MomentRetrievalModel"
_c.model.model_dim = _c.model_dim
_c.model.chunk_num = _c.chunk_num
_c.model.use_negative = False
_c.model.text_config = CN()
_c.model.text_config.input_dim = 300
_c.model.text_config.hidden_dim = _c.model_dim
_c.model.video_config = CN()
_c.model.video_config.input_dim = 500
_c.model.video_config.hidden_dim = _c.model_dim
_c.model.video_config.kernel_size = [1]
_c.model.video_config.stride = [1]
_c.model.video_config.padding = [0]
_c.model.audio_config = CN()
_c.model.audio_config.input_dim = _c.audio_dim




# Train Config
_c.train = CN()
_c.train.batch_size = 64
_c.train.max_epoch = 50
_c.train.display_interval = 50
_c.train.saved_path = "checkpoints/anet_other"


# Test Config
_c.test = CN()
_c.test.batch_size = 1
_c.test.type_list = ["moment_retrieval"]
_c.test.args_list = [{"top_n": 5, "thresh": 0.5, "by_frame": False, "display_interval": 100}]

# Dataset Config
_c.dataset = CN()
_c.dataset.name = "AnetCaption"
_c.dataset.feature_path = "/home/data/activity-c3d"
_c.dataset.vocab_path = "/home/glove_model.bin"
_c.dataset.audio_path = "/home/data/wavs_16k"
_c.dataset.data_path_template = "/home/new_{}_data.json"
_c.dataset.max_frame_num = _c.max_frame_num
_c.dataset.max_word_num = _c.max_word_num
_c.dataset.max_audio_num = _c.max_audio_num
_c.dataset.audio_dim = _c.audio_dim
_c.dataset.frame_dim = _c.model.video_config.input_dim
_c.dataset.word_dim = _c.model.text_config.input_dim

# Optimizer Config
_c.optimizer = CN()
_c.optimizer.lr = 5e-4
_c.optimizer.warmup_updates = 800
_c.optimizer.warmup_init_lr = 1e-7
_c.optimizer.weight_decay = 1e-5
_c.optimizer.loss_config = CN()
_c.optimizer.loss_config.boundary = 1
_c.optimizer.loss_config.inside = 6
_c.optimizer.loss_config.norm_bias = 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _c.clone()
