import pandas as pd
import numpy as np
from QIG import (
    QIG
)

obs_data = pd.read_csv('~sachs_exp/sachs_obs_data.csv')

sachsQIG = QIG(obs_data, [[]])
sachsQIGcim = sachsQIG.linsolv()

np.savez(
        "sachs_obs_QIG_results.npz",
        data=sachsQIG.data,
        targets=sachsQIG.targets,
        skeleton=sachsQIG.skeleton[3],
        coordinates=sachsQIG.coordinates,
        BIC_sim=sachsQIG.BIC_sim,
        BIC_cim=sachsQIG.BIC_cim,
        leaf_targets=sachsQIG.leaf_targets,
        leaf_ints=sachsQIG.leaf_ints,
        inequalities=sachsQIG.inequalities,
        learned_cim=sachsQIGcim.x
)

# With interventions at {0: obs, 1: Akt, 2: PKC, 3: PIP2, 4: MeK, 5: PIP3}
int_data1 = pd.read_csv('~sachs_exp/2-sachs_Akt.csv')
int_data2 = pd.read_csv('~sachs_exp/3-sachs_PKC.csv')
int_data3 = pd.read_csv('~sachs_exp/4-sachs_PIP2.csv')
int_data4 = pd.read_csv('~sachs_exp/5-sachs_MEK.csv')
int_data5 = pd.read_csv('~sachs_exp/6-sachs_PIP3.csv')


datasets = [obs_data, int_data1, int_data2, int_data3, int_data4, int_data5]
ints = [[], [6], [8], [3], [1], [4]]

sachsiQIG = QIG(datasets, ints)

sachsiQIGcim = sachsiQIG.linsolv()

np.savez(
        "sachs_int_QIG_results.npz",
        data=sachsiQIG.data,
        targets=sachsiQIG.targets,
        skeleton=sachsiQIG.skeleton,
        coordinates=sachsiQIG.coordinates,
        BIC_sim=sachsiQIG.BIC_sim,
        BIC_cim=sachsiQIG.BIC_cim,
        leaf_targets=sachsiQIG.leaf_targets,
        leaf_ints=sachsiQIG.leaf_ints,
        inequalities=sachsiQIG.inequalities,
        learned_cim=sachsiQIGcim.x
)
