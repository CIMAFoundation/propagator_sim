#%%
from os import walk 
from scipy.io import loadmat
#%%

for root, dirs, files in walk("/share/propagator_production/propagator_sim/data"):
    for file in files:
        if file.endswith(".mat"):
            _data = loadmat(root + "/" + file)
            if 'M' not in _data:
                continue

            M = _data["M"]
            
            if M.shape != (2000, 2000):
                print(file)
                print(M.shape)
                
# %%
