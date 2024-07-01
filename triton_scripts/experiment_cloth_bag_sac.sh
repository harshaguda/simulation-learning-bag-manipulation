cd ${WRKDIR}/OmniIsaacGymEnvs
export ISAACSIM_PYTHON_EXE="/isaac-sim/python.sh"
$ISAACSIM_PYTHON_EXE -m pip install -e .
$ISAACSIM_PYTHON_EXE -m pip install -e .
cd ${WRKDIR}/cloth-bag-manipulation-learning
$ISAACSIM_PYTHON_EXE scripts/sac_trainer.py task=ClothBagAttach num_envs=8 headless=True
