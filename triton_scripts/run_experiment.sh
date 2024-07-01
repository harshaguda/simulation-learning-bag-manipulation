#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --constraint=ampere
export ACCEPT_EULA=Y
export ISAACSIM_PATH="/isaac-sim"
export ISAACSIM_PYTHON_EXE="/isaac-sim/python.sh"

# run experiment
srun singularity exec -B /m:/m -B /l:/l -B /scratch:/scratch -B /etc/vulkan/icd.d/nvidia_icd.json -B /etc/vulkan/implicit_layer.d/nvidia_layers.json -B /usr/share/glvnd/egl_vendor.d/10_nvidia.json -B ${WRKDIR}/isaac-sim/kit/cache/Kit:/isaac-sim/kit/cache/Kit,${WRKDIR}/isaac-sim/cache/ov:/isaac-sim/cache/ov,${WRKDIR}/isaac-sim/cache/pip:/isaac-sim/cache/pip,${WRKDIR}/isaac-sim/cache/glcache:/isaac-sim/cache/glcache,${WRKDIR}/isaac-sim/cache/computecache:/isaac-sim/cache/computecache,${WRKDIR}/isaac-sim/logs:/isaac-sim/logs,${WRKDIR}/isaac-sim/data:/isaac-sim/data,${WRKDIR}/isaac-sim/documents:/isaac-sim/documents,${WRKDIR}/isaac-sim/kit/logs/Kit/Isaac-Sim:/isaac-sim/kit/logs/Kit/Isaac-Sim,${WRKDIR}/isaac-sim/kit/exts/omni.gpu_foundation/cache/nv_shadercache:/isaac-sim/kit/exts/omni.gpu_foundation/cache/nv_shadercache,${WRKDIR}/isaac-sim/kit/data/Kit/omni.isaac.sim.python.gym/2023.1:/isaac-sim/kit/data/Kit/omni.isaac.sim.python.gym/2023.1,${WRKDIR}/isaac-sim/kit/data/Kit/omni.isaac.sim.python.gym.camera/2023.1:/isaac-sim/kit/data/Kit/omni.isaac.sim.python.gym.camera/2023.1,${WRKDIR}/cloth-bag-manipulation-learning/kits:/isaac-sim/apps --nv ${WRKDIR}/singularity_images/isaac-sim.sif sh ${WRKDIR}/cloth-bag-manipulation-learning/triton_scripts/$1 $2