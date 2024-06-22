num_frames = 40
frame_interval = 3
image_size = (256, 256)

# Define dataset
root = "data root" # Before the first colunm in the csv file, if the path is full path, you can set it to ""
data_path = "csv path"
use_image_transform = False
num_workers = 2

# Define acceleration
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained="bora checkpoint path",
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="your vae path",
)
text_encoder = dict(
    type="t5",
    from_pretrained="your T5 path",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "model checkpoints save path"
wandb = False

epochs = 100
log_every = 306 # based on the batch size, any number is OK actually
ckpt_every = 6120 # based on the batch size
load = None

batch_size = 4
lr = 1e-5
grad_clip = 1.0
