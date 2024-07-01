cd ${WRKDIR}/OmniIsaacGymEnvs
export ISAACSIM_PYTHON_EXE="/isaac-sim/python.sh"
$ISAACSIM_PYTHON_EXE -m pip install -e .
$ISAACSIM_PYTHON_EXE -m pip install -e .
$ISAACSIM_PYTHON_EXE -m pip install shapely
echo $1
cd ${WRKDIR}/cloth-bag-manipulation-learning
$ISAACSIM_PYTHON_EXE scripts/ppo_trainer_headless.py task=ClothBagAttach num_envs=32 headless=True seed=$1
