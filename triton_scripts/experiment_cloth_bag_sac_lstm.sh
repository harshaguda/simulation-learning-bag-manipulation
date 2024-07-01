cd ${WRKDIR}/OmniIsaacGymEnvs
export ISAACSIM_PYTHON_EXE="/isaac-sim/python.sh"
$ISAACSIM_PYTHON_EXE -m pip install -e .
$ISAACSIM_PYTHON_EXE -m pip install -e .
cd ${WRKDIR}/cloth-bag-manipulation-learning
$ISAACSIM_PYTHON_EXE scripts/torch_gymnasium_pendulumnovel_sac_lstm.py task=ClothBagAttach num_envs=2 headless=True
