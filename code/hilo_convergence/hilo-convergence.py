import os
os.chdir('..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.phosphene_model import RectangleImplant, MVGModel
from src.DSE import load_mnist, sample_cn_characters, rand_model_params, fetch_dse
from src.HILO import HILOPatient, patient_from_phi_arr

def run_hilo(patient, num_duels, present_targets, eval_sets, test_every):
    d = patient.d
    xtrain = np.empty((d*2, num_duels), dtype='double') # phi1/phi2 used in duels
    ctrain = np.empty((num_duels), dtype='double') # responses
    losses_sets = [[] for _ in eval_sets]
    
    pbar = tqdm(range(num_duels), unit='duels')
    for idx_duel in pbar:
        if idx_duel == 0:
            xtrain[:, idx_duel] = patient.hilo_acquisition(None, None)
        else:
            xtrain[:, idx_duel] = patient.hilo_acquisition(xtrain[:, :idx_duel], ctrain[:idx_duel])
        phi1 = xtrain[:d, idx_duel]
        phi2 = xtrain[d:, idx_duel]

        target = present_targets[np.random.randint(0, len(present_targets))] # get a random target
        # simulate the duel
        decision, resdict = patient.duel(target, phi1, phi2)
        ctrain[idx_duel] = decision
        # update posterior
        patient.hilo_update_posterior(xtrain[:, :idx_duel+1], ctrain[:idx_duel+1])
        # get the current best guess for true phi
        phi_guess = patient.hilo_identify_best(xtrain[:, :idx_duel+1], ctrain[:idx_duel+1])
        
        # Evaluate on test set
        if idx_duel % test_every == 0 or idx_duel == num_duels-1:
            for set_id,eval_targets in enumerate(eval_sets):
                dse_loss = patient.mismatch_dse.evaluate(x=[eval_targets, \
                    tf.repeat(phi_guess[None, ...], len(eval_targets), axis=0)], y=eval_targets, batch_size=256, verbose=0)
                losses_sets[set_id].append(dse_loss)
            pbar.set_description(f"loss: {dse_loss : .4f}")
    return phi_guess, losses_sets

# setup
version='v2' # version from paper, with bug fixed.
num_patients = 5
hilo_steps = 50
test_every = 5
test_on = 2000
np.random.seed(42)
implant = RectangleImplant()
model = MVGModel(xrange=(-12, 12), yrange=(-12, 12), xystep=0.5).build()
dse = fetch_dse(model, implant, version=version)
(_, _), (mnist_targets, _) = load_mnist(model)
(_, _), (casia_targets, _) = sample_cn_characters(model, "assets/Gnt1.1Test/", 4000)
matlab_dir = 'matlab/'
phis = rand_model_params(num_patients, version=version)
t_mnist = mnist_targets[:test_on]
t_casia = casia_targets[:test_on]

phis_mnist, phis_casia = [], []
losses_mm, losses_mc, losses_cm, losses_cc = [], [], [], []

for phi in phis:
    model, implant = patient_from_phi_arr(phi, model, implant, implant_kwargs={})
    patient = HILOPatient(model, implant, dse=dse, phi_true=phi, matlab_dir=matlab_dir, version=version)
    phi_guess, [losses_1, losses_2] = run_hilo(patient, hilo_steps, mnist_targets, [t_mnist, t_casia], test_every)
    phis_mnist.append(phi_guess)
    losses_mm.append(losses_1); losses_mc.append(losses_2)
    model, implant = patient_from_phi_arr(phi, model, implant, implant_kwargs={})
    patient = HILOPatient(model, implant, dse=dse, phi_true=phi, matlab_dir=matlab_dir, version=version)
    phi_guess, [losses_1, losses_2] = run_hilo(patient, hilo_steps, casia_targets, [t_mnist, t_casia], test_every)
    phis_casia.append(phi_guess)
    losses_cm.append(losses_1); losses_cc.append(losses_2)

import pickle
results = {"phis_mnist": phis_mnist,
           "phis_casia": phis_casia,
           "losses": [losses_mm, losses_mc, losses_cm, losses_cc],
           "hilo_steps": hilo_steps,
           "test_every": test_every,
           "test_on": test_on}
with open("hilo-converge-results.pkl",'wb') as f:
    pickle.dump(results, f)