import os 

for i in range(10) :
    path_dir = './{}'.format(i)
    if not os.path.exists(path_dir) : 
        os.mkdir(path_dir)
    cmd = 'mv {}_img* {}/'.format(i, i)
    os.system(cmd)