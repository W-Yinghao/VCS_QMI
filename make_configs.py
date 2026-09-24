"""Create frozen pilot specifications; server agent implements the consuming CLI."""
from copy import deepcopy
from pathlib import Path
import json
import yaml

ROOT=Path(__file__).resolve().parent
base={
    'schema_version':'vcs_ssl_agent_1.0',
    'run': {'stage':'P3_pilot','method':'vcs_qmi','seed':0,'output_root':'${OUTPUT_ROOT}',
            'resume':None,'overwrite':False},
    'data': {'name':'cifar10','root':'${DATA_ROOT}','download':False,'split':'dev45k_val5k',
             'split_seed':20260924,'val_per_class':500,'manifest':'${MANIFEST_ROOT}/cifar10_dev45k_val5k.json',
             'ssl_labels_accessible':False,'official_test_accessible':False},
    'views': {'count':2,'random_resized_crop':{'size':32,'scale':[0.2,1.0],
              'ratio':[0.75,1.3333333333333333],'interpolation':'bilinear','antialias':True},
              'horizontal_flip_p':0.5,'color_jitter':{'brightness':0.4,'contrast':0.4,'saturation':0.4,'hue':0.1,'p':0.8},
              'grayscale_p':0.2,'gaussian_blur_p':0.0,'solarize_p':0.0,
              'normalize_mean':[0.5,0.5,0.5],'normalize_std':[0.5,0.5,0.5]},
    'model': {'backbone':'resnet18_cifar','weights':None,'h_dim':512,
              'stem':{'kernel_size':3,'stride':1,'padding':1,'bias':False,'maxpool':False},
              'projector':{'hidden_dim':512,'output_dim':128,'hidden_batchnorm':True,
                           'hidden_linear_bias':False,'output_linear_bias':True,'output_batchnorm':False},
              'normalization':{'vcs_and_simclr':'l2','eps':1e-8,'vicreg':'none'},
              'critic':{'enabled':True,'input':'ordered_concat','hidden_dims':[512,512],
                        'activation':'relu','output':'tanh','batchnorm':False,'dropout':0.0,
                        'last_layer_xavier_gain':0.1,'last_layer_bias':0.0}},
    'objective':{'target':'mixture_reference_I_VQ','loss':'negative_J','positive_weight':1.0,'negative_weight':1.0,
                 'training_cs_transform':False,'clip_J':False,'extra_regularizers':[],
                 'simclr_temperature':0.2,'vicreg_weights':{'invariance':25.0,'variance':25.0,'covariance':1.0},
                 'vicreg_variance_eps':1e-4},
    'pairing':{'sampler':'random_nonzero_cyclic_shift','k':1,'unique_shifts':True,
               'allow_self':False,'label_filter':False,'queue':False,'negative_detach':False,
               'rng':'dedicated_cpu_generator','rng_seed_offset':100003},
    'train':{'mode':'joint','epochs':20,'warmup_epochs':2,'batch_size_images':256,
             'drop_last':True,'shuffle':True,'replacement':False,'world_size':1,
             'grad_accumulation_steps':1,'precision':'fp32','allow_tf32':False,'compile':False,
             'num_workers':4,'pin_memory':True,'persistent_workers':False,'grad_clip_norm':None,
             'encoder_projector_forward':'concat_2B','max_steps':None},
    'optimizer':{'name':'adamw','lr':0.001,'betas':[0.9,0.999],'eps':1e-8,
                 'matrix_weight_decay_encoder_projector':0.0001,
                 'weight_decay_bias_norm':0.0,'critic_weight_decay':0.0,'critic_lr_multiplier':1.0},
    'schedule':{'kind':'linear_warmup_cosine','unit':'optimizer_step','min_lr_ratio':0.01,'scale_lr_with_batch':False},
    'evaluation':{'official_test_enabled':False,'checkpoint_rule':'fixed_last_epoch',
                  'knn_epochs':[0,5,10,20],'linear_epochs_of_pretrain':[20],
                  'clean_transform':'to_tensor_then_fixed_normalization_only',
                  'feature':'h_before_projector','freeze_encoder_parameters':True,'freeze_bn_buffers':True,
                  'knn':{'k':200,'temperature':0.1,'normalize_h':True,'query_chunk':256},
                  'linear':{'normalize_h':False,'head':'linear_with_bias','epochs':100,'batch_size':256,
                            'optimizer':'sgd','lr':0.1,'momentum':0.9,'weight_decay':0.0,
                            'schedule':'cosine','min_lr_ratio':0.001,'seed':20260925,
                            'checkpoint_rule':'final_probe_epoch','feature_cache_dtype':'float32'},
                  'critic_validation':{'enabled':True,'augmentation':'same_two_view_distribution_as_train',
                                       'repeats':4,'update_weights':False,'model_mode':'eval','batch_size':256,
                                       'rng_seed':20260926},
                  'spectrum':{'enabled':True,'selection_first_sorted_ids':4096,
                              'features':['h','p_raw','z_l2'],'center':True,'covariance_ddof':1}},
    'logging':{'step_interval':50,'gradient_norm_interval':50,'epoch_jsonl':True,
               'save_initial_checkpoint':True,'checkpoint_epochs':[5,10,20],
               'resume_checkpoint':'last.pt','keep_failure_artifacts':True,'external_logging':False},
    'execution':{'max_concurrent_jobs':1,'max_gpus_per_job':1,'automatic_hyperparameter_search':False,
                 'auto_start_full_200ep':False,'requires_preflight':True},
}
for name,method in [('vcs','vcs_qmi'),('simclr','simclr_matched'),('vicreg','vicreg_matched_128')]:
    c=deepcopy(base);c['run']['method']=method
    if name!='vcs':
        c['model']['critic']['enabled']=False
        c['evaluation']['critic_validation']['enabled']=False
        c['objective']['target']='InfoNCE_training_objective' if name=='simclr' else 'VICReg_loss'
        c['objective']['loss']='nt_xent' if name=='simclr' else 'variance_invariance_covariance'
    (ROOT/'configs'/f'cifar10_pilot_{name}.yaml').write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf-8')
# Full run is a non-executable plan, not a silently enabled config.
full={'status':'PLANNED_NOT_AUTHORIZED_BY_THIS_STARTER',
      'dataset':'cifar10','dev_epochs':200,'warmup_epochs':10,
      'seed_list':[0,1,2],'restart_from_initial_weights':True,
      'requires':['P3_report_review','frozen_selection_rule','baseline_tuning_budget','resource_approval'],
      'official_50k_retrain_and_10k_test':'Separate confirmation protocol after recipe lock.'}
(ROOT/'configs'/'next_stage_plan.yaml').write_text(yaml.safe_dump(full,sort_keys=False,allow_unicode=True),encoding='utf-8')
print('Generated 3 pilot configurations and 1 inactive plan.')
