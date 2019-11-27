#!/bin/bash

#rm -r ../out/ctmatch/gf_* name-out-* name-err-* 
#jbsub -q x86_12h -cores 1+1 -mem 12g -interactive bash
jbsub -q x86_7d -cores 1+1 -mem 128g -out name-out-ctmatch1.txt -err name-err-ctmatch1.txt python train.py -config ../config/config1.yml
jbsub -q x86_7d -cores 1+1 -mem 128g -out name-out-ctmatch2.txt -err name-err-ctmatch2.txt python train.py -config ../config/config2.yml
jbsub -q x86_7d -cores 1+1 -mem 128g -out name-out-ctmatch3.txt -err name-err-ctmatch3.txt python train.py -config ../config/config3.yml
