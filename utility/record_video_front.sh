# (X, Y, Z) cm
# X coordinate is from left to right increasing 
# Y coordinate is from back to front increasing
# Z coordinate is bottom to up increasing
# (0, 0, 0) = Front, mid point of the aquarium
# (0, 14, 13.75)

DURATION=$1
IP_ADDRESS=`/usr/sbin/ifconfig | grep "inet 192" | awk '{print $2}' | awk -F '.' '{print $4}'`
TIMESTAMP=`date '+%Y_%m_%d_%H_%M'`
OUTPUT=record_${IP_ADDRESS}_${TIMESTAMP}.mjpeg
WIDTH=2312
HEIGHT=1786
LENS_POS=9.0

rpicam-vid -f --autofocus-mode manual --lens-position $LENS_POS --width $WIDTH --height $HEIGHT --codec mjpeg -t $DURATION -o $OUTPUT
