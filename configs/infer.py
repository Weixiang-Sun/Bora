num_frames = 40
fps = 24 // 3
image_size = (256, 256)

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    from_pretrained="PRETRAINED_MODEL",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="your vae path",
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="your T5 path",
    model_max_length=120,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
    cfg_channel=3, # or None
)
dtype = "fp16"

# Others
batch_size = 1
seed = 42
prompt_path = "your prompts path"
save_dir = "your output folder path"
