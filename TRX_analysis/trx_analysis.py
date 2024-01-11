
import os
import os.path as op
import sys
import shutil
import zipfile
from time import time

import boto3
from botocore import UNSIGNED
from botocore.client import Config

import nibabel as nib
from dipy.io.streamline import load_tractogram
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import pandas as pd

from AFQ.utils.path import drop_extension, read_json, write_json
import AFQ.utils.streamlines as aus
from memory_profiler import memory_usage
from trx.trx_file_memmap import TrxFile, save

from dipy.stats.analysis import afq_profile, gaussian_weights

import nibabel as nib
import configparser

CP = configparser.ConfigParser()
CP.read_file(open("/my_gscratch/.openneuropermission.txt"))
CP.sections()
aws_access_key = CP.get('openneuro', 'aws_access_key_id')
aws_secret_key = CP.get('openneuro', 'aws_secret_access_key')

scratch_dir = "/my_gscratch/hcp_test/"
hcp_sub_list = np.loadtxt("/my_gscratch/hcp_subs_errs.txt", dtype=int)
subject = hcp_sub_list[int(sys.argv[1])-1]

# Create local filesystem:
base_path = op.join(scratch_dir, f"hcp_pyafq_data", f"sub-{subject}")
os.makedirs(base_path, exist_ok=True)
print(f"base path is {base_path}")

client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key)
clean_trk_fname = (
    f"sub-{subject}/ses-01/sub-{subject}_dwi_space-RASMM_model-CSD"
    f"_desc-prob-afq-clean_tractography")
print("Downloading trk...")
if not op.exists(f"{base_path}/tractography.trk"):
    try:
        client.download_file(
            "open-neurodata",
            f"rokem/hcp1200/afq/{clean_trk_fname}.trk",
            f"{base_path}/tractography.trk")
    except:
        clean_trk_fname = (
            f"sub-{subject}/ses-01/sub-{subject}_dwi_space-RASMM_model-CSD"
            f"_desc-prob-AFQ-clean_tractography")
        client.download_file(
            "open-neurodata",
            f"rokem/hcp1200/afq/{clean_trk_fname}.trk",
            f"{base_path}/tractography.trk")

if not op.exists(f"{base_path}/tractography.json"):
    client.download_file(
        "open-neurodata",
        f"rokem/hcp1200/afq/{clean_trk_fname}.json",
        f"{base_path}/tractography.json")
full_trk_fname = (
    f"sub-{subject}/ses-01/sub-{subject}_dwi_space-RASMM_model-CSD"
    "_desc-prob_tractography")
try: 
    if not op.exists(f"{base_path}/tractography_full.trk"):
        client.download_file(
            "open-neurodata",
            f"rokem/hcp1200/afq/{full_trk_fname}.trk",
            f"{base_path}/tractography_full.trk")
except:
    has_full_tractography = False
else:
    has_full_tractography = True

if not op.exists(f"{base_path}/fa.nii.gz"):
    client.download_file(
        "open-neurodata",
        f"rokem/hcp1200/afq/sub-{subject}/ses-01/sub-{subject}_dwi_model-DTI_FA.nii.gz",
        f"{base_path}/fa.nii.gz")

print("making UIDs...")
# note this does not require the downloaded tractography
# just the json
BUNDLES = ["ATR", "CGC", "CST", "IFO", "ILF", "SLF", "ARC", "UNC",
           "FA", "FP"]
CALLOSUM_BUNDLES = ["AntFrontal", "Motor", "Occipital", "Orbital",
                    "PostParietal", "SupFrontal", "SupParietal",
                    "Temporal"]
bundle_names = BUNDLES + CALLOSUM_BUNDLES
bundle_uids = {}
uid = 1
for name in bundle_names:
    if name in ["FA", "FP"]:
        bundle_uids[name] = uid
        uid += 1
    elif name in CALLOSUM_BUNDLES:
        bundle_uids[name] = uid
        uid += 1
    else:
        for hemi in ['_R', '_L']:
            bundle_uids[name+hemi] = uid
            uid += 1
sidecar_info = read_json(f"{base_path}/tractography.json")
sidecar_info["bundle_ids"] = bundle_uids
to_upload_json_fname = f"{scratch_dir}/jsons/sub-{subject}_dwi_space-RASMM_model-CSD_desc-prob-afq-clean_tractography.json"
write_json(
    to_upload_json_fname,
    sidecar_info)
write_json(
    f"{base_path}/tractography.json",
    sidecar_info)

seg_sft = aus.SegmentedSFT.fromfile(f"{base_path}/tractography.trk")
seg_sft.sft.dtype_dict = { 
    'positions': np.float16, 'offsets': np.uint32,
    'dpv': {}, 'dps': {}}
trx = TrxFile.from_sft(seg_sft.sft, dtype_dict=seg_sft.sft.dtype_dict)
bd_idxs = {}
for key, value in seg_sft.bundle_idxs.items():
    bd_idxs[key] = np.array(value)
trx.groups = bd_idxs
trx_fname = f"{scratch_dir}/trxs/sub-{subject}_dwi_space-RASMM_model-CSD_desc-prob-afq-clean_tractography.trx"
save(
    trx,
    trx_fname,
    compression_standard=zipfile.ZIP_DEFLATED)

err_stuff = {
    "Bundle Name": [],
    "Avg. Absolute Err. (um)": []}
seg_sft_trx = aus.SegmentedSFT.fromfile(
    trx_fname,
    sidecar_file=f"{base_path}/tractography.json")
seg_sft = aus.SegmentedSFT.fromfile(f"{base_path}/tractography.trk")
fa_img = nib.load(f"{base_path}/fa.nii.gz")

for b_name in trx.groups.keys():
    sls32 = seg_sft.get_bundle(b_name).streamlines
    if len(sls32) == 0:
        print(f"Warning: {b_name} skipped")
        continue
    sls16 = seg_sft_trx.get_bundle(b_name).streamlines

    err_abs = 0
    err_c = 0
    for idx in range(len(sls16)):
        mean_dist = np.linalg.norm(sls32[idx]-sls16[idx], axis=1)
        err_abs += np.mean(np.absolute(mean_dist))*1000
        err_c += 1
    err_abs /= err_c
    err_stuff["Bundle Name"].append(b_name)
    err_stuff["Avg. Absolute Err. (um)"].append(err_abs)
    print(f"{b_name} avg. absolute err (um): {err_abs}")

profiles_trk = {}
profiles_trx = {}
def profile(this_seg_sft, scalar_img, profile_dict):
    peak_mem = []
    times = []
    for bundle_name in this_seg_sft.bundle_names:
        this_sl = this_seg_sft.get_bundle(bundle_name).streamlines
        if len(this_sl) != 0:
            peak_mem.append(
                max(memory_usage(proc=\
                    lambda: afq_profile(
                        scalar_img.get_fdata(),
                        this_sl,
                        scalar_img.affine,
                        weights=gaussian_weights(this_sl)))))
            start_time = time()
            profile_dict[bundle_name] = afq_profile(
                scalar_img.get_fdata(),
                this_sl,
                scalar_img.affine,
                weights=gaussian_weights(this_sl))
            times.append(time()-start_time)
    return peak_mem, times

err_stuff["Peak Memory usage TRK (MiB)"],\
    err_stuff["Time TRK (s)"]=\
        profile(seg_sft, fa_img, profiles_trk)
err_stuff["Peak Memory usage TRX (MiB)"],\
    err_stuff["Time TRX (s)"]=\
        profile(seg_sft_trx, fa_img, profiles_trx)

err_stuff["Profile error (%)"] = []
err_stuff["TRK mean FA"] = []
err_stuff["TRX mean FA"] = []
for bundle_name in profiles_trk.keys():
    err_stuff["Profile error (%)"].append(100 * np.mean(np.absolute(
        profiles_trk[bundle_name] - profiles_trx[bundle_name])/profiles_trk[bundle_name]))
    err_stuff["TRK mean FA"].append(np.mean(profiles_trk[bundle_name]))
    err_stuff["TRX mean FA"].append(np.mean(profiles_trx[bundle_name]))

print(err_stuff)
pd.DataFrame(err_stuff).to_csv(f"{scratch_dir}/err_csvs/{subject}.csv", index=False)
del trx
del seg_sft

if has_full_tractography:
    sft = load_tractogram(f"{base_path}/tractography_full.trk", "same", Space.RASMM)
    sft.dtype_dict = {
        'positions': np.float16, 'offsets': np.uint32,
        'dpv': {}, 'dps': {}}
    trx = TrxFile.from_sft(sft, dtype_dict=sft.dtype_dict)
    full_track_trx_fname = f"{scratch_dir}/trxs/sub-{subject}_dwi_space-RASMM_model-CSD_desc-prob_tractography.trx"
    save(
        trx,
        full_track_trx_fname,
        compression_standard=zipfile.ZIP_DEFLATED)

shutil.rmtree(base_path)
print("Completed Calculations...")

if True:
    client.upload_file(
        trx_fname,
        "open-neurodata",
        f"rokem/hcp1200/afq/{clean_trk_fname}.trx")
    os.remove(trx_fname)
    client.upload_file(
        to_upload_json_fname,
        "open-neurodata",
        f"rokem/hcp1200/afq/{clean_trk_fname}.json")
    os.remove(to_upload_json_fname)
    if has_full_tractography:
        client.upload_file(
            full_track_trx_fname,
            "open-neurodata",
            f"rokem/hcp1200/afq/{full_trk_fname}.trx")
        os.remove(full_track_trx_fname)
    print("Upload Complete!")
