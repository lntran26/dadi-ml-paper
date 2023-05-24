import glob

all_inferences = sorted(glob.glob("inference/*bestfits*"))

converged = ''
for fname in all_inferences:
    for line in open(fname, "r"):
        if "Converged" in line:
            converged += fname.split('.')[0].split('_')[1] + ','
print(f"Converged fs from bestfits: {converged}")

not_converged = ''
for fname in all_inferences:
    fs_id = fname.split('.')[0].split('_')[1]
    if fs_id not in converged:
        not_converged += fs_id + ','
print(f"Non converged fs: {not_converged}")

all_out = sorted(glob.glob("outfiles/dadi/*.out"))

converged_out = ''
non_converged_out = ''

for fname in all_out:
    for line in open(fname, "r"):
        if "running" in line:
            fs_id = line.strip().split(' ')[2]
        if "Converged" in line:
            converged_out += fs_id + ','

for fname in all_inferences:
    fs_id = fname.split('.')[0].split('_')[1]
    if fs_id not in converged_out:
        non_converged_out += fs_id + ','

print(f"Non-converged fs from outfiles: {non_converged_out}")