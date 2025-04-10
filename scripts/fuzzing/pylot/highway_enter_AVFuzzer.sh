export TF_CPP_MIN_LOG_LEVEL=1

WORKSPACE=${PWD}
PYLOT_PATH="${WORKSPACE}/ads/systems/Pylot_project"
export PYLOT_HOME="${PYLOT_PATH}"
save_root="${WORKSPACE}/results/MoDitector/data_collection"
export PYLOT_CARLA_HOME="${PYLOT_PATH}/dependencies/CARLA_0.9.10.1"
export PYTHONPATH=$PYLOT_PATH:$PYTHONPATH:$PYLOT_PATH/dependencies:${PYLOT_CARLA_HOME}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:${PYLOT_CARLA_HOME}/PythonAPI/carla/agents

GPU=0
server_config='fuzzing_process_1'
carla_port=$(grep 'port:' ${WORKSPACE}/configs/server_configs/${server_config}.yaml | awk '{print $2}')
echo use carla port ${carla_port}

scenario_name='highway_enter'
agent_name='pylot'

entry_point_target_agent='ads.systems.Pylot.ERDOSRootCauseAgent:ERDOSRootCauseAgent'
ori_config_target_agent=${WORKSPACE}/ads/systems/Pylot/pylot_configs/root_cause.conf
ori_fuzzer_config_file="${WORKSPACE}/configs/main_fuzzer.yaml"


fuzzing_name='AVFuzzer'
entry_point_fuzzer=fuzzer.suites:${fuzzing_name}

config_used=${scenario_name}_${fuzzing_name}
config_target_agent=${WORKSPACE}/configs/pylot_configs/${scenario_name}_${fuzzing_name}.conf
fuzzer_config_file=${WORKSPACE}/configs/${config_used}.yaml
cp ${ori_config_target_agent} ${config_target_agent}
cp ${ori_fuzzer_config_file} ${fuzzer_config_file}



erdos_start_port=19100
sed -i "s#--erdos_start_port=.*#--erdos_start_port=${erdos_start_port}#" ${config_target_agent}


total_repeats=1
run_hour=4

mutator_vehicle_num=10
mutator_walker_num=10
mutator_static_num=0

save_folder=${save_root}/${agent_name}/${scenario_name}/${fuzzing_name}
seed_path=${WORKSPACE}/data/seeds/${scenario_name}.json

sed -i "s#entry_point_target_agent:.*#entry_point_target_agent: ${entry_point_target_agent}#" ${fuzzer_config_file}
sed -i "s#config_target_agent:.*#config_target_agent: ${config_target_agent}#" ${fuzzer_config_file}
sed -i "s#seed_path:.*#seed_path: ${seed_path}#" ${fuzzer_config_file}
sed -i "s#gpu:.*#gpu: ${GPU}#" ${fuzzer_config_file}
sed -i "s#entry_point_fuzzer:.*#entry_point_fuzzer: ${entry_point_fuzzer}#" ${fuzzer_config_file}

sed -i "s#project_root:.*#project_root: ${WORKSPACE}#" ${fuzzer_config_file}
sed -i "s#save_root:.*#save_root: ${save_folder}#" ${fuzzer_config_file}

sed -i "s#--obstacle_detection_model_paths=.*#--obstacle_detection_model_paths=${WORKSPACE}/ads/systems/Pylot_project/dependencies/models/obstacle_detection/faster-rcnn#" ${config_target_agent}
sed -i "s#path_coco_labels=.*#path_coco_labels=${WORKSPACE}/ads/systems/Pylot_project/dependencies/models/pylot.names#" ${config_target_agent}


for run_index in $(seq "1" "$total_repeats"); do
  kill -9 $(ps -ef|grep ${carla_port}$|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')
  kill -9 $(ps -ef|grep ${server_config}|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')
  echo "Current run time: $run_index"
  #CUDA_VISIBLE_DEVICES=$GPU python ${WORKSPACE}/main_fuzzer.py \
  python ${WORKSPACE}/main_fuzzer.py \
  -cn ${config_used} \
  server_configs=$server_config \
  save_root=$save_folder \
  seed_path=$seed_path \
  gpu=$GPU \
  time_limit=$run_hour \
  mutator_vehicle_num=$mutator_vehicle_num \
  mutator_walker_num=$mutator_walker_num \
  mutator_static_num=$mutator_static_num 
  sleep 1
done

rm ${config_target_agent}
rm ${fuzzer_config_file}