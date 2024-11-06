timeout 3h zsh scripts/data_collection/pylot/highway_enter.sh
kill -9 $(ps -ef|grep pylot|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')