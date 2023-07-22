# this file is loaded at ~/.profile
# with adding line . ~/peurakarkotin/.profile

. ~/peurakarkotin/.env

alias ll="ls -lah"
alias peurakarkotin="python3 detect.py 1>> success.log 2>> error.log &"
alias temp="vcgencmd measure_temp"

cd $ROOT_DIRECTORY
