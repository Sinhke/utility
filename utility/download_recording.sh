IP_SUFFIX=$1
OUTDIR=$2

scp pi@192.168.1.${IP_SUFFIX}:/home/pi/record_\*.mjpeg $OUTDIR/
