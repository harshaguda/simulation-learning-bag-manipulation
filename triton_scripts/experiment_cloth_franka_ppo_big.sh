cd ${WRKDIR}/OmniIsaacGymEnvs # path to the OmniIsaacGymEnvs folder
export ISAACSIM_PYTHON_EXE="/isaac-sim/python.sh"
$ISAACSIM_PYTHON_EXE -m pip install -e .
$ISAACSIM_PYTHON_EXE -m pip install -e .
cd ${WRKDIR}/cloth-bag-manipulation-learning
$ISAACSIM_PYTHON_EXE scripts/ppo_big_headless.py task=ClothBagAttachFranka num_envs=32 headless=True seed=$1
