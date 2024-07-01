cd ${WRKDIR}/OmniIsaacGymEnvs
export ISAACSIM_PYTHON_EXE="/isaac-sim/python.sh"
$ISAACSIM_PYTHON_EXE -m pip install -e .
$ISAACSIM_PYTHON_EXE -m pip install -e .
cd ${WRKDIR}/cloth-bag-manipulation-learning
$ISAACSIM_PYTHON_EXE scripts/ppo_vision_trainer.py task=ClothBagCamera num_envs=2 headless=True seed=$1
