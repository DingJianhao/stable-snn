import subprocess

# for id in [
#         # 'vgg11DLI_ATFGSM[0.1000,2.00]',
#         # 'vgg11DLI_ATFGSM[0.5000,2.00]',
#     # 'vgg11DLI_ATFGSM[1.0000,2.00]',
#     #'vgg11DLI_ATFGSM[5.0000,2.00]'
    
#     # 'vgg5DLI_ATFGSM[0.0000,4.00]v2',
#     # 'vgg5DLI_ATFGSM[0.5000,4.00]v2',
#     # 'vgg5DLI_ATFGSM[1.0000,4.00]v2',
#     # 'vgg5DLI_ATFGSM[2.0000,4.00]v2',
    
#     # 'vgg5DLI_ATFGSM[0.0000,4.00,0.00]v2',
    
#     'vgg5DLI_ATFGSM[3.0000,4.00,0.50]v2',
#     'vgg5DLI_ATFGSM[4.0000,4.00,0.50]v2',
    
#     # 'vgg5LIF_ATFGSM[0.0000,4.00,0.00]v2',
#     # 'vgg5LIF_ATFGSM[1.0000,4.00,0.50]v2',
# ]: 
#         subprocess.call(['python', 
#                     'main_test.py', 
#                     '--id=%s' % id, '--model=vgg5', '--data=cifar10',
#                          # '--neuron_type=LIFSpike',
#                 '--neuron_type=DLIFSpike',
#                     '--config=standard', '-dev=0'])



# for id in [
#         # 'wrn16DLI_ATFGSM[1.0000,4.00,0.50]v2',
#         # 'wrn16DLI_ATFGSM[0.0000,4.00,0.50]v2',
#         # 'wrn16DLI_Gaus[1.0000,8.00,0.50]v2',
#     # 'wrn16DLI_RAWv2',
#     'wrn16DLI_Gaus[0.0000,8.00,0.50]v2',
# ]: 
#         subprocess.call(['python', 
#                     'main_test.py', 
#                     '--id=%s' % id, '--model=wrn16', '--data=cifar100',
#                         #  '--neuron_type=LIFSpike',
#                 '--neuron_type=DLIFSpike',
#                     '--config=standard', '-dev=1'])





# for id in [
#     'vgg11LIF_RAWv2',
# ]: 
#         subprocess.call(['python', 
#                     'main_test.py', 
#                     '--id=%s' % id, '--model=vgg11', '--data=cifar10',
#                          '--neuron_type=LIFSpike',
#                     '--config=standard', '-dev=0'])
# for id in [
#     'wrn16LIF_RAWv2',
# ]: 
#         subprocess.call(['python', 
#                     'main_test.py', 
#                     '--id=%s' % id, '--model=wrn16', '--data=cifar10',
#                          '--neuron_type=LIFSpike',
#                     '--config=standard', '-dev=1'])
# for id in [
#         # 'wrn16DLI_ATFGSM[1.0000,4.00,0.50]v2',
#         # 'wrn16DLI_Gaus[1.0000,8.00,0.50]v2',
#         # 'wrn16DLI_ATFGSM[0.0000,4.00,0.50]v2'
#     # 'wrn16DLI_RAWv2',
#     'wrn16DLI_Gaus[0.0000,8.00,0.50]v2'
# ]: 
#         subprocess.call(['python', 
#                     'main_test.py', 
#                     '--id=%s' % id, '--model=wrn16', '--data=cifar10',
#                         #  '--neuron_type=LIFSpike',
#                 '--neuron_type=DLIFSpike',
#                     '--config=standard', '-dev=0'])
# for id in [
#         # 'vgg11DLI_ATFGSM[1.0000,4.00,0.50]v2-sd1000',
#     # 'vgg11DLI_ATFGSM[0.0000,4.00,0.50]v2-sd1000'
#     # 'vgg11DLI_ATFGSM[0.0000,4.00,0.50]v2-sd77629',
#     # 'vgg11DLI_ATFGSM[1.0000,4.00,0.50]v2-sd77629'，
#     # 'vgg11DLI_RAWv2',
#     'vgg11DLI_Gaus[0.0000,8.00,0.50]v2',
# ]: 
#         subprocess.call(['python', 
#                     'main_test.py', 
#                     '--id=%s' % id, '--model=vgg11', '--data=cifar100',
#                 '--neuron_type=DLIFSpike',
#                     '--config=standard', '-dev=1'])

        
        
        
        
        
        
# for id in [
#         # 'vgg11DLI_RATFGSM[1.0000,4.00,0.50]v2-sd1000',
#     # 'vgg11DLI_RATFGSM[0.0000,4.00,0.50]v2-sd1000'
#     # 'vgg11DLI_Gaus[0.0000,8.00,0.50]v2',
#     # 'vgg11DLI_RAWv2',
# ]: 
#         subprocess.call(['python', 
#                     'main_test.py', 
#                     '--id=%s' % id, '--model=vgg11', '--data=cifar10',
#                 '--neuron_type=DLIFSpike',
#                     '--config=standard', '-dev=1'])




# for id in [
#     # 'vgg11DLI_RATFGSM[1.0000,4.00,0.50]v2-sd1000',
#     # 'vgg11DLI_RATFGSM[0.0000,4.00,0.50]v2-sd1000',
#     # 'vgg11DLI_ATFGSM[1.0000,4.00,0.50]v2',
#     # 'vgg11DLI_ATFGSM[0.0000,4.00,0.50]v2',
    
# ]: 
#         subprocess.call(['python', 
#                     'main_test.py', 
#                     '--id=%s' % id, '--model=vgg11', '--data=cifar10',
#                 '--neuron_type=DLIFSpike',
#                     '--config=pgd-eps', '-dev=1'])
        
        
for id in [
    'vgg11DLI_Gaus[1.0000,8.00,0.50]v2-sd89',
    # 'vgg11DLI_Gaus[0.0000,8.00,0.50]v2-sd89',
]: 
        subprocess.call(['python', 
                    'main_test.py', 
                    '--id=%s' % id, '--model=vgg11', '--data=cifar10',
                '--neuron_type=DLIFSpike',
                    '--config=standard', '-dev=1'])