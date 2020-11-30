#! /bin/bash  
for i in *_tumor_*.pkl; do mv $i ${i/*_tumor_/tumor_}; done
