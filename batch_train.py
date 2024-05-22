import subprocess

# ablation
# for alpha in [4]: # 0   0.5   1 2
#     subprocess.call(['python', 
#                 'main_train-v2.py', 
#                 '--model=vgg5', 
#                 '--dataset=cifar10', 
#                 '--alpha=%f' % alpha, 
#                      '--kappa=0.5',
#                      # '--neuron_type=LIFSpike',
#                 '--neuron_type=DLIFSpike', 
#                 '--training_mode=AT_FGSM', '-dev=0'])


configs = [
    # dev data      arch.    training    alpha   regbeta 
    (0,  'cifar10', 'vgg11', 'Gaussian',  1.0,  0.0005),
    (0,  'cifar10', 'vgg11',  'AT_FGSM',  1.0,  0.0005),
    (0,  'cifar10', 'vgg11',  'AT_FGSM',  0.0,  0.0005),
    (0,  'cifar10', 'vgg11', 'RAT_FGSM',  1.0,   0.001),
    (0,  'cifar10', 'vgg11', 'RAT_FGSM',  0.0,   0.001),
    
    (0,  'cifar100', 'vgg11', 'Gaussian',  1.0,  0.0005),
    (0,  'cifar100', 'vgg11',  'AT_FGSM',  1.0,  0.0005),
    (0,  'cifar100', 'vgg11',  'AT_FGSM',  0.0,  0.0005),
    (0,  'cifar100', 'vgg11', 'RAT_FGSM',  1.0,   0.001),
    (0,  'cifar100', 'vgg11', 'RAT_FGSM',  0.0,   0.001),
    
    (0,  'cifar10', 'wrn16', 'Gaussian',  1.0,  0.0005),
    (0,  'cifar10', 'wrn16',  'AT_FGSM',  1.0,  0.0005),
    (0,  'cifar10', 'wrn16',  'AT_FGSM',  0.0,  0.0005),
    (0,  'cifar10', 'wrn16', 'RAT_FGSM',  1.0,   0.004),
    (0,  'cifar10', 'wrn16', 'RAT_FGSM',  0.0,   0.004),
    
    (0,  'cifar100', 'wrn16', 'Gaussian',  1.0,  0.0005),
    (0,  'cifar100', 'wrn16',  'AT_FGSM',  1.0,  0.0005),
    (0,  'cifar100', 'wrn16',  'AT_FGSM',  0.0,  0.0005),
    (0,  'cifar100', 'wrn16', 'RAT_FGSM',  1.0,   0.004),
    (0,  'cifar100', 'wrn16', 'RAT_FGSM',  0.0,   0.004),
    ]

# configs_ = [ # dev 0 gp1
#     (0,  'cifar10', 'vgg11', 'RAT_FGSM',  1.0,   0.001),
#     (0,  'cifar10', 'vgg11', 'RAT_FGSM',  0.0,   0.001),
#     (0,  'cifar10', 'vgg11', 'Gaussian',  1.0,  0.0005),
#     (0,  'cifar10', 'vgg11',  'AT_FGSM',  1.0,  0.0005),
#     (0,  'cifar10', 'vgg11',  'AT_FGSM',  0.0,  0.0005),
# ]

# configs_ = [ # dev 0 gp2
#     (0,  'cifar10', 'wrn16', 'RAT_FGSM',  1.0,   0.004),
#     (0,  'cifar10', 'wrn16', 'RAT_FGSM',  0.0,   0.004),
#     (0,  'cifar10', 'wrn16', 'Gaussian',  1.0,  0.0005),
#     (0,  'cifar10', 'wrn16',  'AT_FGSM',  1.0,  0.0005),
#     (0,  'cifar10', 'wrn16',  'AT_FGSM',  0.0,  0.0005),
# ]

# configs_ = [ # dev 1 gp1
#     (1,  'cifar100', 'vgg11', 'RAT_FGSM',  1.0,   0.001),
#     (1,  'cifar100', 'vgg11', 'RAT_FGSM',  0.0,   0.001),
#     (1,  'cifar100', 'vgg11', 'Gaussian',  1.0,  0.0005),
#     (1,  'cifar100', 'vgg11',  'AT_FGSM',  1.0,  0.0005),
#     (1,  'cifar100', 'vgg11',  'AT_FGSM',  0.0,  0.0005),
# ]

# configs_ = [ # dev 1 gp2
#     (1,  'cifar100', 'wrn16', 'RAT_FGSM',  1.0,   0.004),
#     (1,  'cifar100', 'wrn16', 'RAT_FGSM',  0.0,   0.004),
#     (1,  'cifar100', 'wrn16', 'Gaussian',  1.0,  0.0005),
#     (1,  'cifar100', 'wrn16',  'AT_FGSM',  1.0,  0.0005),
#     (1,  'cifar100', 'wrn16',  'AT_FGSM',  0.0,  0.0005),
# ]

# for config in configs_:
#     subprocess.call(['python', 
#             'main_train-v2.py', 
#             '--model=%s' % config[2], 
#             '--dataset=%s' % config[1], 
#             '--alpha=%f' % config[4], '--beta=%f' % config[5],
#             '--training_mode=%s' % config[3], '-dev=%d' % config[0]])
    
    
    
    
    
    
# new_configs_ = [ # dev1 gp1
#     # (0,  'cifar10', 'vgg11', 'RAT_FGSM',  1.0,   0.0005),
#     (0,  'cifar10', 'vgg11', 'RAT_FGSM',  0.0,   0.0005),
# ]

# new_configs_ = [ # dev1 gp1
#     # (1,  'cifar100', 'vgg11', 'AT_FGSM',  1.0,  0.0005),
#     # (1,  'cifar100', 'vgg11',  'AT_FGSM',  0.0,  0.0005),
# ]

# for config in new_configs_:
#     subprocess.call(['python', 
#             'main_train-v2.py', '--seed=1000', '--suffix=-sd1000',
#             '--model=%s' % config[2], 
#             '--dataset=%s' % config[1], 
#             '--alpha=%f' % config[4], '--beta=%f' % config[5],
#             '--training_mode=%s' % config[3], '-dev=%d' % config[0]])


# new_new_configs_ = [ # dev1 gp1
#     # (0,  'cifar100', 'vgg11',  'AT_FGSM',  0.0,  0.0005, 'DLIFSpike', 77629), #ok
#     # (0,  'cifar10', 'vgg11', 'Gaussian',  0.0,  0.0005, 'DLIFSpike', 42), # trok
#     # (0,  'cifar100', 'vgg11', 'Gaussian',  0.0,  0.0005, 'DLIFSpike', 42), # ep60
#     # (0,  'cifar10', 'vgg11', 'Raw',  0.0,  0.0005, 'LIFSpike', 42), #
    
#     # (1,  'cifar100', 'vgg11',  'AT_FGSM',  1.0,  0.0005, 'DLIFSpike', 77629),#ok
#     # (1,  'cifar100', 'vgg11', 'Raw',  0.0,  0.0005, 'LIFSpike', 42), # trok
#     # (1,  'cifar10', 'vgg11', 'Raw',  0.0,  0.0005, 'DLIFSpike', 42), # trok
#     # (1,  'cifar100', 'vgg11', 'Raw',  0.0,  0.0005, 'DLIFSpike', 42), # trok
    
    
#     # (0,  'cifar10', 'wrn16', 'Gaussian',  0.0,  0.0005, 'DLIFSpike', 42), # trok
#     # (0,  'cifar100', 'wrn16', 'Gaussian',  0.0,  0.0005, 'DLIFSpike', 42), # trok
#     # (0,  'cifar10', 'wrn16', 'Raw',  0.0,  0.0005, 'LIFSpike', 42), # ep30
    
#     # (1,  'cifar100', 'wrn16', 'Raw',  0.0,  0.0005, 'LIFSpike', 42),#ok
#     # (1,  'cifar10', 'wrn16', 'Raw',  0.0,  0.0005, 'DLIFSpike', 42),#ok
#     # (1,  'cifar100', 'wrn16', 'Raw',  0.0,  0.0005, 'DLIFSpike', 42),#ok
# ]

# for config in new_new_configs_:
#     l = '-j=8'
#     if config[7] != 42:
#         l = '--suffix=-sd77629'
#     subprocess.call(['python', 
#             'main_train-v2.py', 
#             '--model=%s' % config[2],  '--seed=%d' % config[7], l,
#             '--dataset=%s' % config[1], 
#             '--alpha=%f' % config[4], '--beta=%f' % config[5], '--neuron_type=%s' % config[6], 
#             '--training_mode=%s' % config[3], '-dev=%d' % config[0]])



new_new_new_configs_ = [ # dev1 gp1
    # (0,  'cifar10', 'vgg11', 'Gaussian',  0.0,  0.0005, 'DLIFSpike', 2024666888),
    (1,  'cifar10', 'vgg11', 'Gaussian',  1.0,  0.0005, 'DLIFSpike', 2024666888),
]

for config in new_new_new_configs_:
    l = '-j=8'
    if config[7] != 42:
        l = '--suffix=-sd2024666888'
    subprocess.call(['python', 
            'main_train-v2.py', 
            '--model=%s' % config[2],  '--seed=%d' % config[7], l,
            '--dataset=%s' % config[1], 
            '--alpha=%f' % config[4], '--beta=%f' % config[5], '--neuron_type=%s' % config[6], 
            '--training_mode=%s' % config[3], '-dev=%d' % config[0]])