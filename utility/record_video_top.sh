# (X, Y, Z) cm
# X coordinate is from left to right increasing 
# Y coordinate is from back to front increasing
# Z coordinate is bottom to up increasing
# (0, 0, 0) = Front, mid point of the aquarium
# (0, -6, 38) 

DURATION=$1
IP_ADDRESS=`/usr/sbin/ifconfig | grep "inet 192" | awk '{print $2}' | awk -F '.' '{print $4}'`
TIMESTAMP=`date '+%Y_%m_%d_%H_%M'`
OUTPUT=vide_record_${IP_ADDRESS}_${TIMESTAMP}.mjpeg
WIDTH=2312
HEIGHT=1786
LENS_POS=7.5


rpicam-vid -f --width $WIDTH --height $HEIGHT --codec mjpeg -t $DURATION -o $OUTPUT #--autofocus-mode manual --lens-position $LENS_POS 
