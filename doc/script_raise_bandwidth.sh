sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.core.rmem_default=65536
sysctl -w net.core.wmem_default=65536
sysctl -w net.ipv4.tcp_rmem='40960 390000 1870000'
sysctl -w net.ipv4.tcp_wmem='40960 390000 1870000'
sysctl -w net.ipv4.tcp_mem='1870000 1870000 1870000'
sysctl -w net.ipv4.route.flush=1

#sysctl -w net.ipv4.tcp_no_metrics_save=1
#sysctl -w net.ipv4.tcp_moderate_rcvbuf=1

