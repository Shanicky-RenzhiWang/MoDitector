export TF_CPP_MIN_LOG_LEVEL=1

WORKSPACE=/home/${USER}/workspace
PYLOT_PATH="${WORKSPACE}/pylot"
export PYTHONPATH=$PYLOT_PATH:$PYTHONPATH
project_root=${WORKSPACE}/MoDitector
save_root="${WORKSPACE}/results/MoDitector/data_collection1"

GPU=0
server_config='fuzzing_process_2'

# port=$(sed -n 's/.*port: \([0-9]*\).*/\1/p' ${project_root}/configs/server_configs/${server_config}.yaml)
# echo $port

# nohup zsh -c "$PYLOT_HOME/scripts/run_simulator.sh $port" > /tmp/carla.log 2>&1 &
# echo "Carla simulator loading..."
# sleep 5s |pv -t

export scenario_name='intersection_left'
agent_name='pylot'
config_used='root_cause_pylot'

entry_point_target_agent='ads.systems.Pylot.ERDOSRootCauseAgent:ERDOSRootCauseAgent'
# entry_point_target_agent='ads.systems.Pylot.ERDOSAgent:ERDOSAgent'
# config_target_agent=${project_root}/ads/systems/Pylot/pylot_configs/perfect_detection_pylot.conf
config_target_agent=${project_root}/ads/systems/Pylot/pylot_configs/root_cause_gt_perception.conf

fuzzer_config_file="${WORKSPACE}/MoDitector/configs/root_cause_pylot.yaml"
# fuzzing_name='RootCauseFuzzer'
# entry_point_fuzzer='fuzzer.suites:RootCauseFuzzer'

fuzzing_name='ReplayRunner'
entry_point_fuzzer='fuzzer.suites:ReplayRunner'


# fuzzer_config_file="${WORKSPACE}/MoDitector/configs/${config_used}.yaml"
# fuzzing_name='RandomFuzzer'
# entry_point_fuzzer='fuzzer.suites:BehAVExplor'



total_repeats=1
run_hour=4

mutator_vehicle_num=20
mutator_walker_num=0
mutator_static_num=0

save_folder=${save_root}/${agent_name}/${scenario_name}/RootCauseFuzzer
seed_path=${project_root}/data/seeds/${scenario_name}.json

sed -i "s#entry_point_target_agent:.*#entry_point_target_agent: ${entry_point_target_agent}#" ${fuzzer_config_file}
sed -i "s#config_target_agent:.*#config_target_agent: ${config_target_agent}#" ${fuzzer_config_file}
sed -i "s#seed_path:.*#seed_path: ${seed_path}#" ${fuzzer_config_file}
sed -i "s#gpu:.*#gpu: ${GPU}#" ${fuzzer_config_file}

# for run_index in $(seq "1" "$total_repeats"); do
#   echo "Current run time: $run_index"
#   #CUDA_VISIBLE_DEVICES=$GPU python ${project_root}/main_fuzzer.py \
#   python ${project_root}/main_fuzzer.py \
#   -cn ${config_used} \
#   server_configs=$server_config \
#   save_root=$save_folder \
#   seed_path=$seed_path \
#   gpu=$GPU \
#   time_limit=$run_hour \
#   mutator_vehicle_num=$mutator_vehicle_num \
#   mutator_walker_num=$mutator_walker_num \
#   mutator_static_num=$mutator_static_num
#   sleep 1
# done

CUDA_VISIBLE_DEVICES=$GPU python ${project_root}/replay.py \
  -cn ${config_used} \
  server_configs=$server_config \
  save_root=$save_folder \
  seed_path=$seed_path \
  gpu=$GPU \
  time_limit=$run_hour \
  mutator_vehicle_num=$mutator_vehicle_num \
  mutator_walker_num=$mutator_walker_num \
  mutator_static_num=$mutator_static_num