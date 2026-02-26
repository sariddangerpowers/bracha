import torch
import numpy as np
from tqdm import tqdm
from utils import log_string


class EvaluationTransformer:
    """
    Handles evaluation on a given POI dataset for the Transformer model.

    Same metrics as the original (MAP, recall@n) but without hidden state
    management — each batch is self-contained.
    """

    def __init__(self, dataset, dataloader, user_count, trainer, setting, log):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.trainer = trainer
        self.setting = setting
        self._log = log

    def evaluate(self):
        self.dataset.reset()

        with torch.no_grad():
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            average_precision = 0.

            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)

            eval_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                            desc='  Evaluating', leave=False)
            for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in eval_bar:
                active_users = active_users.squeeze(0)

                # Track reset_count to skip already-evaluated users (same logic as original)
                for j, reset in enumerate(reset_h):
                    if reset:
                        reset_count[active_users[j]] += 1

                # squeeze dim=0 removes the DataLoader batch dim only
                x = x.squeeze(0).to(self.setting.device)
                t = t.squeeze(0).to(self.setting.device)
                t_slot = t_slot.squeeze(0).to(self.setting.device)
                s = s.squeeze(0).to(self.setting.device)

                y = y.squeeze(0)
                y_t = y_t.squeeze(0).to(self.setting.device)
                y_t_slot = y_t_slot.squeeze(0).to(self.setting.device)
                y_s = y_s.squeeze(0).to(self.setting.device)

                # When sequence_length=1, squeeze collapses the seq dim too. Restore it.
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                    t = t.unsqueeze(0)
                    t_slot = t_slot.unsqueeze(0)
                    y = y.unsqueeze(0)
                    y_t = y_t.unsqueeze(0)
                    y_t_slot = y_t_slot.unsqueeze(0)
                if s.dim() == 2:  # s has an extra coord dim
                    s = s.unsqueeze(0)
                    y_s = y_s.unsqueeze(0)
                active_users = active_users.to(self.setting.device)

                # evaluate (no hidden state):
                out = self.trainer.evaluate(x, t, t_slot, s, y_t, y_t_slot, y_s, active_users)
                # out: (batch, seq, loc_count) on GPU
                # y: (seq, batch) on CPU

                y_gpu = y.to(self.setting.device)  # (seq, batch)
                y_t_idx = y_gpu.t()  # (batch, seq)

                # Recall: top-10 indices via GPU topk
                _, topk = torch.topk(out, 10, dim=-1)  # (batch, seq, 10)
                target_exp = y_t_idx.unsqueeze(-1)  # (batch, seq, 1)
                hits = (topk == target_exp)  # (batch, seq, 10)
                hit1 = hits[:, :, :1].any(dim=-1).float()   # (batch, seq)
                hit5 = hits[:, :, :5].any(dim=-1).float()
                hit10 = hits.any(dim=-1).float()

                # MAP: rank = number of logits greater than target + 1
                target_logit = out.gather(-1, target_exp)  # (batch, seq, 1)
                rank = (out > target_logit).sum(dim=-1) + 1  # (batch, seq)
                prec = (1.0 / rank.float())  # (batch, seq)

                # Move to CPU for per-user accumulation
                hit1_cpu = hit1.cpu()
                hit5_cpu = hit5.cpu()
                hit10_cpu = hit10.cpu()
                prec_cpu = prec.cpu()
                active_users_cpu = active_users.cpu()

                for j in range(self.setting.batch_size):
                    if reset_count[active_users_cpu[j]] > 1:
                        continue
                    seq_len_actual = y.size(0)
                    u_iter_cnt[active_users_cpu[j]] += seq_len_actual
                    u_recall1[active_users_cpu[j]] += hit1_cpu[j].sum().item()
                    u_recall5[active_users_cpu[j]] += hit5_cpu[j].sum().item()
                    u_recall10[active_users_cpu[j]] += hit10_cpu[j].sum().item()
                    u_average_precision[active_users_cpu[j]] += prec_cpu[j].sum().item()

            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                average_precision += u_average_precision[j]

                if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0:
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1',
                          formatter.format(u_recall1[j] / u_iter_cnt[j]), 'MAP',
                          formatter.format(u_average_precision[j] / u_iter_cnt[j]), sep='\t')

            log_string(self._log, 'recall@1: ' + formatter.format(recall1 / iter_cnt))
            log_string(self._log, 'recall@5: ' + formatter.format(recall5 / iter_cnt))
            log_string(self._log, 'recall@10: ' + formatter.format(recall10 / iter_cnt))
            log_string(self._log, 'MAP: ' + formatter.format(average_precision / iter_cnt))
            print('predictions:', iter_cnt)
