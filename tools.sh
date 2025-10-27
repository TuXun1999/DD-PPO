parse_var_arg() {
    local arg=$1
    if [[ $arg == *"="* ]]; then
        var_name="${arg%%=*}"
        var_value="${arg#*=}"
        return 0
    else
        return 1
    fi
}

# Log functions
log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

log_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $1"
}