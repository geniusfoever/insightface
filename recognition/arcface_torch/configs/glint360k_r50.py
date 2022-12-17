from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = False
config.output = "experiment/r50"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False
config.save_all_states=True
config.rec = r"E:\dataset\glint\imgs"
config.rec_id=0
config.num_classes = 73382
config.num_image = 3000000
config.num_epoch = 20
config.val_root=r"E:\dataset\lfw\validation"
config.warmup_epoch = 0
config.val_targets = []#['validation']#['lfw', 'cfp_fp', "agedb_30"]
config.num_workers=6
config.init_last_layer=True