import copy
import numpy as np
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import traceback

from confidence.dataset import ListDataset
from utils import so3, torus
from utils.molecules_utils import get_symmetry_rmsd
from utils.sampling import randomize_position, sampling
from utils.diffusion_utils import get_t_schedule


def loss_function_ddp(tr_pred, rot_pred, tor_pred, sidechain_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, backbone_weight=0, sidechain_weight=0, apply_mean=True, no_torsion=False):
    ## the original DiffDock uses DataParallel and DataListLoader to pass in data in the form of a list
    ## thus this code is written in the anticipation that the data will be a list
    ## this loss_function should be rewritten to handle a HeteroDataBatch from DataLoader
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = data.tr_score # torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score.cpu()) ** 2 * tr_sigma.cpu() ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component
    rot_score = data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
    rot_loss = (((rot_pred.cpu() - rot_score.cpu()) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score.cpu() / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(np.concatenate(data.tor_sigma_edge))
        tor_score = data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float()
        tor_loss = ((tor_pred.cpu() - tor_score.cpu()) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score.cpu() ** 2 / tor_score_norm2)).detach()
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index =  data['ligand'].batch[data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = data.num_graphs 
            t_l, t_b_l, c = torch.zeros(num_graphs, device='cpu'), torch.zeros(num_graphs, device='cpu'), torch.zeros(num_graphs, device='cpu')
            c.index_add_(0, index.cpu(), torch.ones(tor_loss.shape, device='cpu'))
            c = c + 0.0001
            t_l.index_add_(0, index.cpu(), tor_loss)
            t_b_l.index_add_(0, index.cpu(), tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if backbone_weight > 0:
        backbone_vecs = data['receptor'].side_chain_vecs 
        backbone_vecs = backbone_vecs[:, 4:]
        backbone_pred = sidechain_pred[:, 4:]

        backbone_base_loss = (backbone_vecs ** 2).detach().mean(dim=1) + 0.0001
        backbone_loss = ((backbone_pred.cpu() - backbone_vecs) ** 2).mean(dim=1) / backbone_base_loss.mean()
        backbone_base_loss = backbone_base_loss / backbone_base_loss.mean()
        if apply_mean:
            backbone_loss, backbone_base_loss = backbone_loss.mean() * torch.ones(1, dtype=torch.float), backbone_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = data['receptor'].batch
            num_graphs = data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(backbone_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, backbone_loss)
            s_b_l.index_add_(0, index, backbone_base_loss)
            backbone_loss, backbone_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            backbone_loss, backbone_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            backbone_loss, backbone_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if sidechain_weight > 0:
        sidechain_vecs = data['receptor'].side_chain_vecs
        chi_angles = sidechain_vecs[:, :4].to(device)
        chi_pred = sidechain_pred[:, :4].to(device)

        chi_pred = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_pred)
        chi_angles = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_angles)

        difference = torch.abs(chi_pred - chi_angles)
        difference = torch.min(difference, 1 - difference) # angles are circular and 360 degrees = 1

        sidechain_base_loss = (chi_angles ** 2).detach().mean(dim=1) + 0.0001
        sidechain_loss = (difference ** 2).mean(dim=1) / sidechain_base_loss.mean()
        sidechain_base_loss = sidechain_base_loss / sidechain_base_loss.mean()
        if apply_mean:
            sidechain_loss, sidechain_base_loss = \
                sidechain_loss.mean().cpu() * torch.ones(1, dtype=torch.float), \
                sidechain_base_loss.mean().cpu() * torch.ones(1, dtype=torch.float)
        else:
            index = data['receptor'].batch
            num_graphs = data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(sidechain_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, sidechain_loss.cpu())
            s_b_l.index_add_(0, index, sidechain_base_loss.cpu())
            sidechain_loss, sidechain_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            sidechain_loss, sidechain_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            sidechain_loss, sidechain_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(
                len(rot_loss), dtype=torch.float)

    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + sidechain_loss * sidechain_weight + backbone_loss * backbone_weight
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), backbone_loss.detach(), sidechain_loss.detach(), \
           tr_base_loss, rot_base_loss, tor_base_loss, backbone_base_loss, sidechain_base_loss

def loss_function(tr_pred, rot_pred, tor_pred, sidechain_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, backbone_weight=0, sidechain_weight=0, apply_mean=True, no_torsion=False):
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score.cpu()) ** 2 * tr_sigma.cpu() ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component
    rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
    rot_loss = (((rot_pred.cpu() - rot_score.cpu()) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score.cpu() / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge))
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float()
        tor_loss = ((tor_pred.cpu() - tor_score.cpu()) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score.cpu() ** 2 / tor_score_norm2)).detach()
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if device.type == 'cuda' else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if backbone_weight > 0:
        backbone_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data], dim=0) if device.type == 'cuda' else data['receptor'].side_chain_vecs
        backbone_vecs = backbone_vecs[:, 4:]
        backbone_pred = sidechain_pred[:, 4:]

        backbone_base_loss = (backbone_vecs ** 2).detach().mean(dim=1) + 0.0001
        backbone_loss = ((backbone_pred.cpu() - backbone_vecs) ** 2).mean(dim=1) / backbone_base_loss.mean()
        backbone_base_loss = backbone_base_loss / backbone_base_loss.mean()
        if apply_mean:
            backbone_loss, backbone_base_loss = backbone_loss.mean() * torch.ones(1, dtype=torch.float), backbone_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)], dim=0).long() if device.type == 'cuda' else data['receptor'].batch
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(backbone_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, backbone_loss)
            s_b_l.index_add_(0, index, backbone_base_loss)
            backbone_loss, backbone_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            backbone_loss, backbone_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            backbone_loss, backbone_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if sidechain_weight > 0:
        sidechain_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data],
                                  dim=0) if device.type == 'cuda' else data['receptor'].side_chain_vecs
        chi_angles = sidechain_vecs[:, :4].to(device)
        chi_pred = sidechain_pred[:, :4].to(device)

        chi_pred = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_pred)
        chi_angles = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_angles)

        difference = torch.abs(chi_pred - chi_angles)
        difference = torch.min(difference, 1 - difference) # angles are circular and 360 degrees = 1

        sidechain_base_loss = (chi_angles ** 2).detach().mean(dim=1) + 0.0001
        sidechain_loss = (difference ** 2).mean(dim=1) / sidechain_base_loss.mean()
        sidechain_base_loss = sidechain_base_loss / sidechain_base_loss.mean()
        if apply_mean:
            sidechain_loss, sidechain_base_loss = \
                sidechain_loss.mean().cpu() * torch.ones(1, dtype=torch.float), \
                sidechain_base_loss.mean().cpu() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)],
                              dim=0).long() if device.type == 'cuda' else data['receptor'].batch
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(sidechain_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, sidechain_loss.cpu())
            s_b_l.index_add_(0, index, sidechain_base_loss.cpu())
            sidechain_loss, sidechain_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            sidechain_loss, sidechain_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            sidechain_loss, sidechain_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(
                len(rot_loss), dtype=torch.float)

    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + sidechain_loss * sidechain_weight + backbone_loss * backbone_weight
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), backbone_loss.detach(), sidechain_loss.detach(), \
           tr_base_loss, rot_base_loss, tor_base_loss, backbone_base_loss, sidechain_base_loss


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum().cpu() if self.unpooled_metrics else v.cpu()
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights):
    model.train()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'])

    #for data in tqdm(loader, total=len(loader)):
    num_nan = 0
    for data_idx, data in enumerate(loader):
        had_error = False
        if not data:
            print('Idx ' + str(data_idx) + ' failed preprocess', flush=True)
            continue
        if data['receptor'].pos.shape[0] == 0:
            print(data['name'][0] + ' was cropped!', flush=True)
            continue
        if data_idx % 10 == 0:
            print('Train idx ' + str(data_idx) + ', receptor size ' + str(data['receptor'].pos.shape[0]) + ' residues', flush=True)
        if isinstance(data, list):
            ## is data is passed in as list -> then the model is a DataParallel model
            if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
                print("Skipping batch of size 1 since otherwise batchnorm would not work.")
                continue
            data = [d.to(device) for d in data] if device.type == 'cuda' else data
        else:
            ## else it is a regular model/DistributedDataParallel model
            data = data.to(device)
        optimizer.zero_grad()

        try:
            tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, device=device)
            loss = loss_tuple[0]

            if torch.any(torch.isnan(loss)):
                names = [d.name for d in data] if isinstance(data, list) else data.name
                print("Nan loss, skipping batch with complexes " + str(names) + ' (idx ' + str(data_idx) + ')', flush=True)
                loss = loss.zero_()
                had_error = True
                num_nan += 1
                #continue

            loss.backward()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch ' + str(data_idx), flush=True)

                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                had_error = True
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch ' + str(data_idx), flush=True)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                had_error = True
            else:
                #raise e
                print(e, flush=True)
                print(traceback.format_exc(), flush=True)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                had_error = True
        if had_error:
            optimizer.zero_grad()
        optimizer.step()
        if ema_weights is not None: ema_weights.update(model.parameters())
        if not had_error:
            meter.add([loss.cpu().detach(), *loss_tuple[1:]])

    out = meter.summary()
    out["num_nan"] = num_nan
    return out


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss',
                          'num_nan'],
                         unpooled_metrics=True)
    num_nan = 0
    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
             'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss',
             'num_nan'],
            unpooled_metrics=True, intervals=10)
    for data_idx, data in enumerate(loader):
        has_error = False
        if data_idx % 10 == 0:
            print('Test idx ' + str(data_idx) + ', receptor size ' + str(data['receptor'].pos.shape[0]) + ' residues', flush=True)
        if not data or data['receptor'].pos.shape[0] == 0:
            continue
        try:
            if isinstance(data, list):
                ## is data is passed in as list -> then the model is a DataParallel model
                if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
                    print("Skipping batch of size 1 since otherwise batchnorm would not work.")
                    continue
                data = [d.to(device) for d in data] if device.type == 'cuda' else data
            else:
                ## else it is a regular model/DistributedDataParallel model
                data = data.to(device)
            with torch.no_grad():
                tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            if loss_tuple is None or torch.any(torch.isnan(loss_tuple[0])):
                has_error = True
                loss_tuple[0].zero_()
                num_nan += 1
                print('Nan detected evaluating complexes ' + str(data.name) + ' (batch idx ' + str(data_idx) + ')')
            else:
                meter.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]])

            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor = [torch.cat([data[i].complex_t[noise_type] for i in range(len(data))]) for
                                                              noise_type in ['tr', 'rot', 'tor']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                meter_all.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]],
                    [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr,
                     sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr])                    
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                print(e)
                continue

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    out["num_nan"] = num_nan
    return out


def inference_epoch_fix(model, loader, device, t_to_sigma, args):
    model.eval()
    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule
    rmsds, min_rmsds = [], []
    centroids, min_centroids = [], []

    for idx, orig_complex_graph in enumerate(loader):
        print('Inference idx ' + str(idx), flush=True)
        if idx >= args.num_inference_complexes:
            break
        if not orig_complex_graph or orig_complex_graph['receptor'].pos.shape[0] == 0:
            continue
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.inference_samples)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)
        data_list = [d.to(device) for d in data_list]

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, model=model.module if device.type == 'cuda' and not (args.no_parallel or args.DDP) else model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args,
                                                         t_schedule=t_schedule,
                                                         temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot, args.temp_sampling_tor] if args.inf_temp else 1.0,
                                                         temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor] if args.inf_temp else 0.0 ,
                                                         temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot, args.temp_sigma_data_tor] if args.inf_temp else 0.5)
            except Exception as e:
                failed_convergence_counter += 1
                if failed_convergence_counter > 5:
                    print('failed 5 times - skipping the complex')
                    break
                print(f"Exception while running inference on complex: {idx}", e)
                print(traceback.format_exc())
        if failed_convergence_counter > 5:
            rmsds.extend([100] * args.inference_samples)
            min_rmsds.append(100)
            continue

        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph[
                                                         'ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
        # if len(orig_complex_graph['ligand'].orig_pos.shape) == 3:
        #     orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        if len(orig_complex_graph['ligand'].orig_pos.shape) == 2:
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[None, :, :]
        try:
            orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[:, filterHs] - orig_complex_graph.original_center.cpu().numpy()
        except Exception as e:
            print("problem with orig_pos which is of shape:", orig_complex_graph['ligand'].orig_pos.shape, e)
            continue
        mol = RemoveAllHs(orig_complex_graph.mol[0])
        complex_rmsds = []
        complex_centroids = []
        for i in range(len(orig_ligand_pos)):
            try:
                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], [l for l in ligand_pos])
            except Exception as e:
                print("Using non corrected RMSD because of the error:", e)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))
            complex_rmsds.append(rmsd)
            centroid = np.sqrt(((ligand_pos.mean(axis=1) - orig_ligand_pos[i].mean(axis=0)) ** 2).sum(axis=1))
            complex_centroids.append(centroid)
        complex_rmsds = np.asarray(complex_rmsds)
        rmsd = np.min(complex_rmsds, axis=0)
        rmsds.extend([r for r in rmsd])
        min_rmsds.append(rmsd.min(axis=0))
        complex_centroids = np.asarray(complex_centroids)
        centroid = np.min(complex_centroids, axis=0)
        centroids.extend([c for c in centroid])
        min_centroids.append(centroid.min(axis=0))

    rmsds = np.array(rmsds)
    min_rmsds = np.array(min_rmsds)
    centroids = np.array(centroids)
    min_centroids = np.array(min_centroids)
    losses = {'rmsd_median': np.median(rmsds),
              'centroid_median': np.median(centroids),
              'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
              'centroid_lt2': (100 * (centroids < 2).sum() / len(centroids)),
              'centroid_lt5': (100 * (centroids < 5).sum() / len(centroids)),
              'min_rmsds_lt2': (100 * (min_rmsds < 2).sum() / len(min_rmsds)),
              'min_rmsds_lt5': (100 * (min_rmsds < 5).sum() / len(min_rmsds)),
              'min_centroid_lt2': (100 * (min_centroids < 2).sum() / len(min_centroids)),
              'min_centroid_lt5': (100 * (min_centroids < 5).sum() / len(min_centroids)),
              }
    return losses
