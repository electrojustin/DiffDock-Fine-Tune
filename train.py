import copy
import math
import setproctitle
import os
import shutil
from functools import partial
import cProfile
import time
import datetime
import wandb
import torch
import torch.distributed as dist
import pstats
import io
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from numpy import mean
from torch.utils.data.distributed import DistributedSampler
from datasets.dataloader import DataLoader, DataListLoader
from datasets.lazy_pdbbind import LazyPDBBindSet
from socket import gethostname
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.pdbbind import NoiseTransform
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
import yaml
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, t_to_sigma_individual
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, loss_function_ddp, inference_epoch_fix
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage



def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, pdbbind_loader):

    if args.DDP or args.no_parallel:
        loss_fn = partial(loss_function_ddp, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                    tor_weight=args.tor_weight, no_torsion=args.no_torsion, backbone_weight=args.backbone_loss_weight,
                    sidechain_weight=args.sidechain_loss_weight)
    else:
        loss_fn = partial(loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                    tor_weight=args.tor_weight, no_torsion=args.no_torsion, backbone_weight=args.backbone_loss_weight,
                    sidechain_weight=args.sidechain_loss_weight)

    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_val_secondary_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    if args.inference_earlystop_avg_infsteps > 0:
        running_val_inference_metric = []
        running_best_val_loss = math.inf
        running_best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
        running_best_val_secondary_value = math.inf if args.inference_earlystop_goal == 'min' else 0
        running_best_epoch = 0
        running_best_val_inferece_epoch = 0

    freeze_params = 0
    scheduler_mode = args.inference_earlystop_goal if args.val_inference_freq is not None else 'min'
    if args.scheduler == 'layer_linear_warmup':
        freeze_params = args.warmup_dur * (args.num_conv_layers + 2) - 1
        print("Freezing some parameters until epoch {}".format(freeze_params))

    # if args.restart_dir:
    epoch_iter = args.n_epochs_range
    print(f"Doing training...Epochs {list(epoch_iter)[0]}–{list(epoch_iter)[-1]}")
    # else:
    #     print("Starting training...")
    #     epoch_iter = range(args.n_epochs)

    # epoch 0 calculate train/val inference performance
    logs = {}
    epoch=list(epoch_iter)[0]-1
    if args.train_inference_freq != None:
        inf_metrics = inference_epoch_fix(model, train_loader, args.device, t_to_sigma, args)
        print("Epoch {}: Train inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
        logs.update({'traininf_' + k: v for k, v in inf_metrics.items()}, step=epoch)

    if args.val_inference_freq != None:
        inf_metrics = inference_epoch_fix(model, val_loader, args.device, t_to_sigma, args)
        print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
        logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch)

    if args.pdbbind_inference_freq != None:
        inf_metrics = inference_epoch_fix(model, pdbbind_loader, args.device, t_to_sigma, args)
        print("Epoch {}: PDBBind inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
        logs.update({'pdbbindinf_' + k: v for k, v in inf_metrics.items()}, step=epoch)

    if args.wandb:
        if args.DDP:
            if args.local_rank==0:
                print("log wandb: rank 0")
                wandb.log(logs, step=epoch+1)
        else:
            wandb.log(logs, step=epoch+1)

    for epoch in epoch_iter:
        if epoch % 5 == 0: print("Run name: ", args.run_name)

        if args.scheduler == 'layer_linear_warmup' and (epoch+1) % args.warmup_dur == 0:
            step = (epoch+1) // args.warmup_dur
            if step < args.num_conv_layers + 2:
                print("New unfreezing step")
                optimizer, scheduler = get_optimizer_and_scheduler(args, model, step=step, scheduler_mode=scheduler_mode)
            elif step == args.num_conv_layers + 2:
                print("Unfreezing all parameters")
                optimizer, scheduler = get_optimizer_and_scheduler(args, model, step=step, scheduler_mode=scheduler_mode)
                ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
        elif args.scheduler == 'linear_warmup' and epoch == args.warmup_dur:
            print("Moving to plateu scheduler")
            optimizer, scheduler = get_optimizer_and_scheduler(args, model, step=1, scheduler_mode=scheduler_mode,
                                                               optimizer=optimizer)

        logs = {}
        train_losses = train_epoch(model, train_loader, optimizer, args.device, t_to_sigma, loss_fn, ema_weights if epoch > freeze_params else None)
        # number of tdqm batches = len(train_dataset) / (args.batch_size)
        
        print("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   sc {:.4f}  lr {:.4f}"
              .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                      train_losses['tor_loss'], train_losses['sidechain_loss'], optimizer.param_groups[0]['lr']))

        if epoch > freeze_params:
            ema_weights.store(model.parameters())
            if args.use_ema: ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
        val_losses = test_epoch(model, val_loader, args.device, t_to_sigma, loss_fn, args.test_sigma_intervals)
        print("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   sc {:.4f}"
              .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss'], val_losses['sidechain_loss']))

        if args.train_inference_freq != None and epoch % args.train_inference_freq == 0:
            inf_metrics = inference_epoch_fix(model, train_loader, args.device, t_to_sigma, args)
            print("Epoch {}: Train inference rmsd_median {:.3f} rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f} centroid_median {:.3f} centroid_lt2 {:.3f} centroid_lt5 {:.3f} min_centroid_lt2 {:.3f} min_centroid_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsd_median'], inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5'], inf_metrics['centroid_median'], inf_metrics['centroid_lt2'], inf_metrics['centroid_lt5'], inf_metrics['min_centroid_lt2'], inf_metrics['min_centroid_lt5']))
            logs.update({'traininf_' + k: v for k, v in inf_metrics.items()}, step=epoch)

        if args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0:
            inf_metrics = inference_epoch_fix(model, val_loader, args.device, t_to_sigma, args)
            print("Epoch {}: Val inference rmsd_median {:.3f} rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f} centroid_median {:.3f} centroid_lt2 {:.3f} centroid_lt5 {:.3f} min_centroid_lt2 {:.3f} min_centroid_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsd_median'], inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5'], inf_metrics['centroid_median'], inf_metrics['centroid_lt2'], inf_metrics['centroid_lt5'], inf_metrics['min_centroid_lt2'], inf_metrics['min_centroid_lt5']))
            logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch)

        if args.pdbbind_inference_freq != None and (epoch + 1) % args.pdbbind_inference_freq == 0:
            inf_metrics = inference_epoch_fix(model, pdbbind_loader, args.device, t_to_sigma, args)
            print("Epoch {}: PDBBind inference rmsd_median {:.3f} rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f} centroid_median {:.3f} centroid_lt2 {:.3f} centroid_lt5 {:.3f} min_centroid_lt2 {:.3f} min_centroid_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsd_median'], inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5'], inf_metrics['centroid_median'], inf_metrics['centroid_lt2'], inf_metrics['centroid_lt5'], inf_metrics['min_centroid_lt2'], inf_metrics['min_centroid_lt5']))
            logs.update({'pdbbindinf_' + k: v for k, v in inf_metrics.items()}, step=epoch)

        if epoch > freeze_params:
            if not args.use_ema: ema_weights.copy_to(model.parameters())
            ema_state_dict = copy.deepcopy(model.module.state_dict() if device.type == 'cuda' and not (args.no_parallel or args.DDP) else model.state_dict())
            ema_weights.restore(model.parameters())

        if args.wandb:
            if args.DDP:
                if args.local_rank==0:
                    print("log wandb: rank 0")
                    print(train_losses)
                    print(val_losses)
                    logs.update({'train_' + k: v for k, v in train_losses.items()})
                    logs.update({'val_' + k: v for k, v in val_losses.items()})
                    logs['current_lr'] = optimizer.param_groups[0]['lr']
                    wandb.log(logs, step=epoch + 1)
            else:
                logs.update({'train_' + k: v for k, v in train_losses.items()})
                logs.update({'val_' + k: v for k, v in val_losses.items()})
                logs['current_lr'] = optimizer.param_groups[0]['lr']
                wandb.log(logs, step=epoch + 1)
                
        state_dict = model.module.state_dict() if device.type == 'cuda' and not (args.no_parallel or args.DDP) else model.state_dict()
        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value) and (not args.DDP or args.rank == 0):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'earlystop_metric': f"{args.inference_earlystop_goal}_{args.inference_earlystop_metric}"
            }, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            if epoch > freeze_params:
                torch.save({
                    'epoch': epoch,
                    'model': ema_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'ema_weights': ema_weights.state_dict(),
                    'earlystop_metric': f"{args.inference_earlystop_goal}_{args.inference_earlystop_metric}"
                }, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))

        if args.inference_earlystop_avg_infsteps > 0 and args.inference_earlystop_metric in logs.keys():
            if args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0:
                running_val_inference_metric.append(logs[args.inference_earlystop_metric])
            print("logs", running_val_inference_metric)
            if (args.inference_earlystop_goal == 'min' and mean(running_val_inference_metric[-args.inference_earlystop_avg_infsteps:],axis=0) <= running_best_val_inference_value or
                    args.inference_earlystop_goal == 'max' and mean(running_val_inference_metric[-args.inference_earlystop_avg_infsteps:],axis=0) >= running_best_val_inference_value) and (not args.DDP or args.rank == 0):
                running_best_val_inference_value = mean(running_val_inference_metric[-args.inference_earlystop_avg_infsteps:],axis=0)
                running_best_val_inference_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'earlystop_metric': f"avg_{args.inference_earlystop_goal}_{args.inference_earlystop_metric}"
                }, os.path.join(run_dir, 'running_best_inference_epoch_model.pt'))
                if epoch > freeze_params:
                    torch.save({
                        'epoch': epoch,
                        'model': ema_state_dict,
                        'optimizer': optimizer.state_dict(),
                        'ema_weights': ema_weights.state_dict(),
                        'earlystop_metric': f"avg_{args.inference_earlystop_goal}_{args.inference_earlystop_metric}"
                    }, os.path.join(run_dir, 'running_ema_best_inference_epoch_model.pt'))


        if args.inference_secondary_metric is not None and args.inference_secondary_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_secondary_metric] <= best_val_secondary_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_secondary_metric] >= best_val_secondary_value) and (not args.DDP or args.rank == 0):
            best_val_secondary_value = logs[args.inference_secondary_metric]
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_secondary_epoch_model.pt'))

        if val_losses['loss'] <= best_val_loss and (not args.DDP or args.rank == 0):
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
            if epoch > freeze_params:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))

        if args.save_model_freq is not None and (epoch + 1) % args.save_model_freq == 0:
            shutil.copyfile(os.path.join(run_dir, 'best_model.pt'),
                            os.path.join(run_dir, f'epoch{epoch+1}_best_model.pt'))

        if scheduler:
            if epoch < freeze_params or (args.scheduler == 'linear_warmup' and epoch < args.warmup_dur):
                scheduler.step()
            elif args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses['loss'])

        if not args.DDP or args.rank == 0:
            torch.save({
                'epoch': epoch,
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'ema_weights': ema_weights.state_dict(),
            }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))



def main_function():
    args = parse_train_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    local_rank=None
    if args.DDP:
        world_size    = int(os.environ["WORLD_SIZE"])
        rank          = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)
        
        print(f"Rank {rank}: Starting init_process_group", flush=True)
        setup(rank, world_size)
        print(f"Rank {rank}: Finished init_process_group", flush=True)

        if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
        print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}", flush=True)
        device = torch.device(f'cuda:{local_rank}')
        args.world_size=world_size
        args.rank=rank
        args.local_rank=local_rank
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.wandb and args.entity:
        wandb_args = {
            'entity':args.entity,
            'settings':wandb.Settings(start_method="fork"),
            'project':args.project,
            'name':args.run_name,
            'group':args.group,
            'config':args,
        }
        if args.restart_dir and args.wandb_id: 
            print(f"resume wandb run: {args.wandb_id}")
            wandb_args.update({
                "id":args.wandb_id,
                "resume":"must"
            })
        if args.DDP:
            ## wandb has issues spawning multiple runs when doing DDP..resolve this by only tracking rank0
            # wandb_args.update({
            #     'group':"DDP",
            #     'job_type':"worker"
            # })
            if args.local_rank==0:
                print("Log DDP only from rank0")
                wandb.init(**wandb_args)
        else:
            wandb.init(**wandb_args)

    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader, _ = construct_loader(args, t_to_sigma, device)

    transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                               all_atom=args.all_atoms, alpha=args.sampling_alpha, beta=args.sampling_beta,
                               include_miscellaneous_atoms=False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                               crop_beyond_cutoff=args.crop_beyond)
    
    if args.pdbbind_inference_freq:
        pdbbind_common_args = {'transform': transform, 'limit_complexes': args.limit_complexes,
                        'chain_cutoff': args.chain_cutoff, 'receptor_radius': args.receptor_radius,
                        'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                        'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                        'matching': not args.no_torsion, 'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                        'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                        'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                        'knn_only_graph': False if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                        'include_miscellaneous_atoms': False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                        'matching_tries': args.matching_tries}
        pdbbind_dataset = LazyPDBBindSet(ligand_file='fixed_ligand', \
            cache_path=args.pdbbind_inf_cache_path, \
            split_path=args.pdbbind_inf_split_path, \
            keep_original=True, \
            esm_embeddings_path=args.pdbbind_inf_esm_embeddings_path,\
            root=args.pdbbind_inf_root, \
            protein_file='protein', require_ligand=True, max_receptor_size=500, **pdbbind_common_args)
        if args.DDP:
            pdbbind_loader = DataLoader(prefetch_factor=args.dataloader_prefetch_factor, dataset=pdbbind_dataset, batch_size=1, num_workers=args.num_dataloader_workers, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last, sampler=DistributedSampler(pdbbind_dataset), collate_fn=lambda batch: [x for x in batch if x is not None], worker_init_fn=lambda worker_id: setproctitle.setproctitle('pdb_dataloader_'+str(worker_id)))
        else:
            pdbbind_loader = DataListLoader(prefetch_factor=args.dataloader_prefetch_factor, dataset=pdbbind_dataset, batch_size=1, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)
    else:
        pdbbind_loader = None

    model = get_model(args, device, t_to_sigma=t_to_sigma, no_parallel=args.no_parallel)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
    ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)

    if args.restart_dir:
        try:
            print(f'{args.restart_dir}/{args.restart_ckpt}.pt')
            chkpt = torch.load(f'{args.restart_dir}/{args.restart_ckpt}.pt', map_location=torch.device('cpu'))
            if args.restart_lr is not None: chkpt['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(chkpt['optimizer'])
            model.load_state_dict(chkpt['model'], strict=True)
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(chkpt['ema_weights'], device=device)
            print("Restarting from epoch", chkpt['epoch'])
            assert args.n_epochs > chkpt['epoch']
            args.n_epochs_range = range(chkpt['epoch'], args.n_epochs)
        except Exception as e:
            print("Exception", e)
            chkpt = torch.load(f'{args.restart_dir}/{args.restart_ckpt}.pt', map_location=torch.device('cpu'))
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                model.module.load_state_dict(chkpt, strict=True)
            else:
                model.load_state_dict(chkpt, strict=True)
            # model.load_state_dict(dict, strict=True)
            args.n_epochs_range = range(0, args.n_epochs)
            print("Due to exception had to take the best epoch and no optimiser")
    elif args.pretrain_dir:
        chkpt = torch.load(f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt', map_location=torch.device('cpu'))
        if 'epoch' in chkpt.keys():
            args.n_epochs_range = range(chkpt['epoch'], args.n_epochs)
        else:
            args.n_epochs_range = range(0, args.n_epochs)
        print(args.n_epochs_range)
        if "model" in chkpt.keys():
            for key in ["optimizer", "ema_weights", "epoch"]:
                if key in chkpt.keys():
                    del chkpt[key]
            try:
                model.module.load_state_dict(chkpt["model"], strict=True)
            except:
                model.load_state_dict(chkpt["model"], strict=True)
        else:
            try:
                model.module.load_state_dict(chkpt, strict=True)
            except:
                model.load_state_dict(chkpt, strict=True)
        print("Using pretrained model", f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt')

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    run_dir = os.path.join(args.log_dir, args.run_name)
    args.device = device

    # record parameters
    if not args.DDP or rank == 0:
        yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
        model_args = copy.deepcopy(args.__dict__)
        if 'DDP' in model_args:
            del model_args['DDP']
        if 'device' in model_args:
            del model_args['device']
        if 'n_epochs_range' in model_args:
            del model_args['n_epochs_range']
        save_yaml_file(yaml_file_name, model_args) 

    if args.cpu_profile:
        profiler = cProfile.Profile(time.process_time)
        profiler.enable()
        train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, pdbbind_loader)
        profiler.disable()
        profiler.print_stats(sort='tottime')
    else:
        train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, pdbbind_loader)

    if args.DDP:
        dist.destroy_process_group()
        if local_rank==0:
            wandb.finish()

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))

if __name__ == '__main__':
    print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()
