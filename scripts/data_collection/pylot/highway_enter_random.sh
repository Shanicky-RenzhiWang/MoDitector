export TF_CPP_MIN_LOG_LEVEL=1

WORKSPACE=/home/${USER}/workspace
PYLOT_PATH="${WORKSPACE}/pylot"
export PYTHONPATH=$PYLOT_PATH:$PYTHONPATH
project_root=${WORKSPACE}/ADSFuzzer
save_root="${WORKSPACE}/results/ADSFuzzer/data_collection"

GPU=0
server_config='fuzzing_process_2'
carla_port=$(grep 'port:' ${project_root}/configs/server_configs/${server_config}.yaml | awk '{print $2}')
echo use carla port ${carla_port}

scenario_name='highway_enter'
agent_name='pylot'

entry_point_target_agent='ads.systems.Pylot.ERDOSRootCauseAgent:ERDOSRootCauseAgent'
ori_config_target_agent=${project_root}/ads/systems/Pylot/pylot_configs/root_cause.conf

ori_fuzzer_config_file="${WORKSPACE}/ADSFuzzer/configs/main_fuzzer.yaml"
ori_fuzzer_config_file="${WORKSPACE}/ADSFuzzer/configs/main_fuzzer.yaml"
fuzzing_name='RandomFuzzer'
entry_point_fuzzer=fuzzer.suites:${fuzzing_name}

config_used=${scenario_name}_${fuzzing_name}
config_target_agent=${project_root}/configs/pylot_configs/${scenario_name}_${fuzzing_name}.conf
fuzzer_config_file=${WORKSPACE}/ADSFuzzer/configs/${config_used}.yaml
cp ${ori_config_target_agent} ${config_target_agent}
cp ${ori_fuzzer_config_file} ${fuzzer_config_file}


erdos_start_port=19200
sed -i "s#--erdos_start_port=.*#--erdos_start_port=${erdos_start_port}#" ${config_target_agent}


total_repeats=1
run_hour=4

mutator_vehicle_num=10
mutator_walker_num=10
mutator_static_num=0

save_folder=${save_root}/${agent_name}/${scenario_name}/${fuzzing_name}
seed_path=${project_root}/data/seeds/${scenario_name}.json

sed -i "s#entry_point_target_agent:.*#entry_point_target_agent: ${entry_point_target_agent}#" ${fuzzer_config_file}
sed -i "s#config_target_agent:.*#config_target_agent: ${config_target_agent}#" ${fuzzer_config_file}
sed -i "s#seed_path:.*#seed_path: ${seed_path}#" ${fuzzer_config_file}
sed -i "s#gpu:.*#gpu: ${GPU}#" ${fuzzer_config_file}
sed -i "s#entry_point_fuzzer:.*#entry_point_fuzzer: ${entry_point_fuzzer}#" ${fuzzer_config_file}


for run_index in $(seq "1" "$total_repeats"); do
  kill -9 $(ps -ef|grep ${carla_port}$|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')
  kill -9 $(ps -ef|grep ${server_config}|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')
  echo "Current run time: $run_index"
  #CUDA_VISIBLE_DEVICES=$GPU python ${project_root}/main_fuzzer.py \
  python ${project_root}/main_fuzzer.py \
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
