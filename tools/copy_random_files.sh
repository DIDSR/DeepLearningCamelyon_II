#! /bin/bash

files=(/home/wzli/Patches_For_Training/training/augtumor2/*.png)
n=${#files[@]}
file_to_retrieve="${files[RANDOM % n]}"
cp $file_to_retrieve /home/wzli/tumor_patch_sample_original

