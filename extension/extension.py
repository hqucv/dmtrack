import os
import shutil

env_path = '/home/cvpr/anaconda3/envs/DMTrack/lib/python3.7/site-packages/'

# change torch.nn.function
package_name = 'torch/nn/'
function_name = 'functional'
f_path = env_path+package_name+function_name+'.py'
if os.path.exists(f_path):
    print('remove ' + f_path)
    os.remove(f_path)
    print('copy to ' + f_path)
    shutil.copyfile(function_name+'.py', f_path)
