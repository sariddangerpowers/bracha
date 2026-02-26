import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
from setting_transformer import SettingTransformer
from trainer_transformer import TransformerTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from evaluation_transformer import EvaluationTransformer
from tqdm import tqdm
from scipy.sparse import coo_matrix

# parse settings
setting = SettingTransformer()
setting.parse()
dir_name = os.path.dirname(setting.log_file)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
setting.log_file = setting.log_file + '_' + timestring
log = open(setting.log_file, 'w')

message = ''.join([f'{k}: {v}\n' for k, v in vars(setting).items()])
log_string(log, message)

# load dataset (same as original)
poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
poi_loader.read(setting.dataset_file)

log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))

dataset = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TRAIN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(), \
    'batch size must be lower than the amount of available users'

# load graphs (same as original)
with open(setting.trans_loc_file, 'rb') as f:
    transition_graph = pickle.load(f)
transition_graph = coo_matrix(transition_graph)

if setting.use_spatial_graph:
    with open(setting.trans_loc_spatial_file, 'rb') as f:
        spatial_graph = pickle.load(f)
    spatial_graph = coo_matrix(spatial_graph)
else:
    spatial_graph = None

if setting.use_graph_user:
    with open(setting.trans_user_file, 'rb') as f:
        friend_graph = pickle.load(f)
    friend_graph = coo_matrix(friend_graph)
else:
    friend_graph = None

with open(setting.trans_interact_file, 'rb') as f:
    interact_graph = pickle.load(f)
interact_graph = csr_matrix(interact_graph)

log_string(log, 'Successfully load graph')

# create transformer trainer
trainer = TransformerTrainer(
    setting.lambda_t, setting.lambda_s, setting.lambda_loc, setting.lambda_user,
    setting.use_weight, transition_graph, spatial_graph, friend_graph,
    setting.use_graph_user, setting.use_spatial_graph, interact_graph)

trainer.prepare(
    poi_loader.locations(), poi_loader.user_count(),
    setting.d_model, setting.n_heads, setting.n_layers, setting.dropout,
    setting.device)

evaluation_test = EvaluationTransformer(
    dataset_test, dataloader_test, poi_loader.user_count(),
    trainer, setting, log)

print(trainer)
log_string(log, str(setting))

# training loop
optimizer = torch.optim.Adam(
    trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20, 40, 60, 80], gamma=0.2)

param_count = trainer.count_parameters()
log_string(log, f'In total: {param_count} trainable parameters')

bar = tqdm(total=setting.epochs)
bar.set_description('Training (Transformer)')

for e in range(setting.epochs):
    dataset.shuffle_users()

    losses = []
    epoch_start = time.time()
    batch_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                     desc=f'  Epoch {e+1}/{setting.epochs}', leave=False)
    for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in batch_bar:
        # squeeze dim=0 removes the DataLoader batch dim only
        x = x.squeeze(0).to(setting.device)
        t = t.squeeze(0).to(setting.device)
        t_slot = t_slot.squeeze(0).to(setting.device)
        s = s.squeeze(0).to(setting.device)

        y = y.squeeze(0).to(setting.device)
        y_t = y_t.squeeze(0).to(setting.device)
        y_t_slot = y_t_slot.squeeze(0).to(setting.device)
        y_s = y_s.squeeze(0).to(setting.device)

        # When sequence_length=1, squeeze collapses the seq dim too. Restore it.
        if x.dim() == 1:
            x = x.unsqueeze(0)
            t = t.unsqueeze(0)
            t_slot = t_slot.unsqueeze(0)
            y = y.unsqueeze(0)
            y_t = y_t.unsqueeze(0)
            y_t_slot = y_t_slot.unsqueeze(0)
        if s.dim() == 2:
            s = s.unsqueeze(0)
            y_s = y_s.unsqueeze(0)
        active_users = active_users.to(setting.device)

        optimizer.zero_grad()
        loss = trainer.loss(x, t, t_slot, s, y, y_t, y_t_slot, y_s, active_users)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        batch_bar.set_postfix(loss=f'{loss.item():.4f}')

    # schedule learning rate:
    scheduler.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One training need {:.2f}s'.format(epoch_end - epoch_start))

    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        log_string(log, f'Epoch: {e + 1}/{setting.epochs}')
        log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')

    should_validate = ((e + 1) in setting.validate_epochs) if setting.validate_epochs else ((e + 1) % setting.validate_epoch == 0)
    if should_validate:
        log_string(log, f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
        evl_start = time.time()
        evaluation_test.evaluate()
        evl_end = time.time()
        log_string(log, 'One evaluate need {:.2f}s'.format(evl_end - evl_start))

bar.close()

# save model weights
weights_dir = './weights'
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)
weight_path = os.path.join(weights_dir, f'transformer_weights_{timestring}.pth')
torch.save(trainer.model.state_dict(), weight_path)
log_string(log, f'Model saved to {weight_path}')
