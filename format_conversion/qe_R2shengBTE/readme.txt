The procedure is used to convert fc2 from format of qe in reciprocal space to that of shengBTE in eV2/A. The procedure follows:
1) transform qe fc2 from reciprocal space to real space. It calls code q2r.x, which is involved in qe software.
2) run python through python qe2phonopy.py scf.in final.fc. The script calls functions in phonopy, and please be sure that it is installed. In addition, qe input is required to input the structure. final.fc is the fc2 from the above step.
