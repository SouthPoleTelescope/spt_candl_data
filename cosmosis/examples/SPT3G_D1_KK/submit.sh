#!/bin/bash
# Submit each DES/KiDS chain as its own 3-node, 192-task SLURM job (64 tasks/node).
# Usage: bash deskids.submit

INIS=(
    #"cosmosis_files/decade_shear/decam-13k.ini"
    #"cosmosis_files/decade_shear/decam-13k_desi.ini"
    #"cosmosis_files/desy3_shear/chain_desy3_hybrid_analysis.ini"
    #"cosmosis_files/desy3_shear/chain_desy3_hybrid_analysis_desi.ini"
    #"cosmosis_files/desy3_3x2pt/des-y3-maglim.ini"
    #"cosmosis_files/desy3_3x2pt/des-y3-maglim_desi.ini"
    #"cosmosis_files/desy3_3x2pt_gmv/des-y3-gmv.ini"
    #"cosmosis_files/desy3_3x2pt_gmv/des-y3-gmv_desi.ini"
    #"cosmosis_files/hscy3_shear/hsc-y3-shear-real.ini"
    #"cosmosis_files/hscy3_shear/hsc-y3-shear-real_desi.ini"
    #"cosmosis_files/hscy3_3x2pt/hscy3_3x2pt.ini"
    #"cosmosis_files/hscy3_3x2pt/hscy3_3x2pt_desi.ini"
    #"cosmosis_files/kids1000_shear/chain_kids1000_hybrid_analysis.ini"
    #"cosmosis_files/kids1000_shear/chain_kids1000_hybrid_analysis_desi.ini"
    #"cosmosis_files/kidslegacy_shear/KiDS-Legacy.ini"
    #"cosmosis_files/kidslegacy_shear/KiDS-Legacy_desi.ini"
    #"cosmosis_files/gmv/gmv.ini"
    #"cosmosis_files/gmv/gmv_desi.ini"
    #"cosmosis_files/desy3_shear_alphak/chain_desy3_hybrid_analysis.ini"
    desy3_3x2pt_gmv_maglim/des-y3-gmv-maglim.ini
    desy3_3x2pt_maglim/des-y3-maglim.ini
)

# Absolute path to this script's own directory (holds the desy3_* ini subdirs).
# Passed to cosmosis so the main ini is found regardless of the run cwd.
KKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p logs

for INI in "${INIS[@]}"; do
    NAME=$(basename "$(dirname "$INI")")

    sbatch <<EOF
#!/bin/bash
#SBATCH -t 168:00:00
#SBATCH --partition=SPT3G
#SBATCH --account=SPT3G
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=3
#SBATCH --cpus-per-task=2
#SBATCH --job-name=${NAME}
#SBATCH --hint=multithread
#SBATCH --output=logs/${NAME}_%j.out
#SBATCH --error=logs/${NAME}_%j.err

module load gcc/9.2.0-sjjvpmg openmpi/4.1.1-d775auy

eval "\$(/lcrc/project/SPT3G/users/ac.yomori/packages/miniconda_v3/bin/conda shell.bash hook)"
conda activate /lcrc/project/SPT3G/users/ac.yomori/packages/miniforge3/envs/env-3.15

export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMP_NUM_THREADS=2
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_btl_vader_single_copy_mechanism=none

source cosmosis-configure
source /lcrc/project/SPT3G/users/ac.yomori/repo/healqest/pipeline/spt3g_20192020/likelihood/cobaya_files2/setup_paths_xover_data.sh

cd /lcrc/project/SPT3G/users/ac.yomori/repo/cosmosis-standard-library

export DIR_CHAIN=/lcrc/project/SPT3G/users/ac.yomori/repo/cosmosis-standard-library/output2/

mpirun -n 192 cosmosis ${KKDIR}/${INI} --mpi
EOF

    echo "Submitted: ${NAME} -> ${INI}"
done
