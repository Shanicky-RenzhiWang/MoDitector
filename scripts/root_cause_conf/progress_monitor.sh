#!/bin/bash

LINES_TO_CHECK=5

log_folder="/home/erdos/workspace/ADSFuzzer/scripts/root_cause_conf/log"
config_file="/home/erdos/workspace/ADSFuzzer/scripts/root_cause_conf/sup/supervisord_project.conf"

program_lst=("highway_exit" "highway_enter" "intersection_left" "intersection_right")
#"roach_collector_intersection_left roach_collector_highway_exit roach_collector_intersection_straight"

restart_sup(){
    local config_file=$1
    local program_name=$2
    # rm ${log_folder}/${program_name}.std.log
    # rm ${log_folder}/${program_name}.out.log
    ~/.local/bin/supervisorctl -c $config_file stop ${program_name}
    sleep 4
    ~/.local/bin/supervisorctl -c $config_file start ${program_name}
    sleep 2
}

# Function to check the log file and restart the program if an error is found
check_log_and_restart() {
    local LOG_FILE=$1  # Pass log file as a parameter to the function
    local PROGRAM_NAME=$2

    # error 1
    if [ -f "$LOG_FILE" ]; then
        if tail -n $LINES_TO_CHECK "$LOG_FILE" | grep -qi "ERROR: Invalid session: no stream"; then
            echo "$(date): Error detected in log. Restarting $PROGRAM_NAME..."
#            rm -f $LOG_FILE
            # ~/.local/bin/supervisorctl -c $config_file stop $PROGRAM_NAME
            # sleep 5
            # ~/.local/bin/supervisorctl -c $config_file start $PROGRAM_NAME
            # sleep 2
            restart_sup ${config_file} ${program_name}
        fi
    else
        echo "$(date): Log file does not exist: $LOG_FILE"
    fi

    # error 2
    if [ -f "$LOG_FILE" ]; then
        if tail -n 1 "$LOG_FILE" | grep -qi "error Connection refused (os error 111)"; then
            echo "$(date): Error detected in log. Restarting $PROGRAM_NAME..."
#            rm -f $LOG_FILE
            # ~/.local/bin/supervisorctl -c $config_file stop $PROGRAM_NAME
            # sleep 5
            # ~/.local/bin/supervisorctl -c $config_file start $PROGRAM_NAME
            # sleep 2
            restart_sup ${config_file} ${program_name}
        fi
    else
        echo "$(date): Log file does not exist: $LOG_FILE"
    fi

    # error 3
#     if [ -f "$LOG_FILE" ]; then
#         if tail -n $LINES_TO_CHECK "$LOG_FILE" | grep -qi "Received an error while connecting to the simulator"; then
#             echo "$(date): Error detected in log. Restarting $PROGRAM_NAME..."
# #            rm -f $LOG_FILE
#             /home/erdos/.local/bin/~/.local/bin/supervisorctl -c $config_file stop $PROGRAM_NAME
#             sleep 5
#             /home/erdos/.local/bin/~/.local/bin/supervisorctl -c $config_file start $PROGRAM_NAME
#             sleep 2
#         fi
#     else
#         echo "$(date): Log file does not exist: $LOG_FILE"
#     fi

    # error 4
    if [ -f "$LOG_FILE" ]; then
        if tail -n $LINES_TO_CHECK "$LOG_FILE" | grep -qi "A sensor took too long to send their data"; then
            echo "$(date): Error detected in log. Restarting $PROGRAM_NAME..."
#            rm -f $LOG_FILE
            # ~/.local/bin/supervisorctl -c $config_file stop $PROGRAM_NAME
            # sleep 5
            # ~/.local/bin/supervisorctl -c $config_file start $PROGRAM_NAME
            # sleep 2
            restart_sup ${config_file} ${program_name}
        fi
    else
        echo "$(date): Log file does not exist: $LOG_FILE"
    fi

    if [ -f "$LOG_FILE" ]; then
        if tail -n $LINES_TO_CHECK "$LOG_FILE" | grep -qi "ERROR: Invalid session: no stream available with id"; then
            echo "$(date): Error detected in log. Restarting $PROGRAM_NAME..."
#            rm -f $LOG_FILE
            # ~/.local/bin/supervisorctl -c $config_file stop $PROGRAM_NAME
            # sleep 5
            # ~/.local/bin/supervisorctl -c $config_file start $PROGRAM_NAME
            # sleep 2
            restart_sup ${config_file} ${program_name}
        fi
    else
        echo "$(date): Log file does not exist: $LOG_FILE"
    fi

    if [ -f "$LOG_FILE" ]; then
        if tail -n $LINES_TO_CHECK "$LOG_FILE" | grep -qi "KeyError"; then
            echo "$(date): Error detected in log. Restarting $PROGRAM_NAME..."
#            rm -f $LOG_FILE
            # ~/.local/bin/supervisorctl -c $config_file stop $PROGRAM_NAME
            # sleep 5
            # ~/.local/bin/supervisorctl -c $config_file start $PROGRAM_NAME
            # sleep 2
            restart_sup ${config_file} ${program_name}
        fi
    else
        echo "$(date): Log file does not exist: $LOG_FILE"
    fi

    # Killed
    # error 6
#    if [ -f "$LOG_FILE" ]; then
#        if tail -n 2 "$LOG_FILE" | grep -qi "Killed"; then
#            echo "$(date): Killed detected in log. Restarting $PROGRAM_NAME..."
##            rm -f $LOG_FILE
#            /home/erdos/.local/bin/~/.local/bin/supervisorctl -c $config_file stop $PROGRAM_NAME
#            sleep 5
#            /home/erdos/.local/bin/~/.local/bin/supervisorctl -c $config_file start $PROGRAM_NAME
#            sleep 2
#        fi
#    else
#        echo "$(date): Log file does not exist: $LOG_FILE"
#    fi
}

# Function to stop all running programs
stop_all_programs() {
    echo "Stopping all programs..."
    for program_name in $program_lst; do
        ~/.local/bin/supervisorctl -c $config_file stop $program_name
    done
}

trap 'echo "Ctrl+C pressed. Exiting..."; stop_all_programs; exit 0' SIGINT


for program_name in $program_lst; do
    #~/.local/bin/supervisorctl -c $config_file restart $program_name
    rm ${log_folder}/${program_name}.out.log*
    rm ${log_folder}/${program_name}.std.log*
    # restart_sup ${config_file} ${program_name}
done

~/.local/bin/supervisord -c $config_file

sleep 5

# Main loop to check both log files every 10 seconds
while true; do
  for program_name in $program_lst; do
    log_file_std="$log_folder/${program_name}.std.log"
    check_log_and_restart "$log_file_std" "$program_name"
  done
  sleep 10
done
