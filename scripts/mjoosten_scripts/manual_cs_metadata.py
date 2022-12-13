
import os
import numpy as np
import matplotlib.pyplot as plt


job_nr = 9
exp_nr = ""
job_type = "CTFFIND"

## J9, _, CTFFIND


if job_type == "CTFFIND":
    ###############################################################################
    # CTFFIND4 

    meta_file = f"metadata_all_exp{exp_nr}.npy"
    if os.path.isfile(meta_file):
        meta = np.load(meta_file, allow_pickle=True)
        has_gt = True
    else:
        has_gt = False
    cs_file = f"/home/tnw-cryosparc/processing/P51/J{job_nr}/exposures_success.cs"
    cs = np.load(cs_file, allow_pickle=True)

    if has_gt:
        gt_defocus = np.zeros((len(cs)))
        subsets_done = []
        idx = 0
        for m in meta:
            subset = m["name"].split("_")[3]
            if subset not in subsets_done:
                subsets_done.append(subset)
                gt_defocus[idx] = np.abs(m["defocus"]*10)
                idx += 1

    cs_defocus = np.zeros((len(cs)))
    cs_fit = np.zeros((len(cs)))
    for cidx, c in enumerate(cs):
        subset = str(c["ctf_plotdata/diag_image_path"]).split("_")[6]
        idx = subsets_done.index(subset) if has_gt else cidx
        cs_defocus[idx] = c["ctf/df1_A"]
        cs_fit[idx] = c["ctf/ctf_fit_to_A"]

    if has_gt:
        fig, ax = plt.subplots(1, figsize=(10,5))
        ax.scatter(gt_defocus, cs_defocus, c=cs_fit, cmap="viridis")
        ax.grid()
        ax.set_xlabel("Ground Truth Defocus (A)")
        ax.set_ylabel("CryoSPARC Defocus (A)")
        ax.set_title("Defocus Comparison")
        fig.colorbar(ax.collections[0], label="CryoSPARC CTF Fit")
        fig.savefig(f"defocus_J{job_nr}.png")
    else:
        fig, ax = plt.subplots(1, figsize=(10,5))
        ax.scatter(cs_defocus, cs_fit, c=cs_fit, cmap="viridis")
        ax.grid()
        ax.set_xlabel("CryoSPARC Defocus (A)")
        ax.set_ylabel("CryoSPARC CTF Fit")
        ax.set_title("Defocus Comparison")
        fig.colorbar(ax.collections[0], label="CryoSPARC CTF Fit")
        fig.savefig(f"defocus_J{job_nr}.png")

elif job_type == "PATCH_CTF":
    ###############################################################################
    # Patch CTF

    meta_file = f"metadata_all_exp{exp_nr}.npy"
    meta = np.load(meta_file, allow_pickle=True)
    cs_file = f"/home/tnw-cryosparc/processing/P51/J{job_nr}/exposures_ctf_estimated.cs"
    cs = np.load(cs_file, allow_pickle=True)

    gt_defocus = np.zeros((len(cs)))
    subsets_done = []
    idx = 0
    for m in meta:
        subset = m["name"].split("_")[3]
        if subset not in subsets_done:
            subsets_done.append(subset)
            gt_defocus[idx] = np.abs(m["defocus"]*10)
            idx += 1

    cs_defocus = np.zeros((len(cs)))
    cs_fit = np.zeros((len(cs)))
    for c in cs:
        subset = str(c["ctf/path"]).split("_")[4]
        idx = subsets_done.index(subset)
        cs_defocus[idx] = c["ctf/df1_A"]
        cs_fit[idx] = c["ctf/ctf_fit_to_A"]

    fig, ax = plt.subplots(1, figsize=(10,5))
    ax.scatter(gt_defocus, cs_defocus, c=cs_fit, cmap="viridis")
    ax.grid()
    ax.set_xlabel("Ground Truth Defocus (A)")
    ax.set_ylabel("CryoSPARC Defocus (A)")
    ax.set_title("Defocus Comparison")
    fig.colorbar(ax.collections[0], label="CryoSPARC CTF Fit")
    fig.savefig(f"defocus_J{job_nr}.png")
