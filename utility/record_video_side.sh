# (X, Y, Z) cm
# X coordinate is from left to right increasing 
# Y coordinate is from back to front increasing
# Z coordinate is bottom to up increasing
# (0, 0, 0) = Front, mid point of the aquarium
# (-1, 15.25, 13)

DURATION=$1
IP_ADDRESS=`ifconfig | grep "inet 192" | awk '{print $2}' | awk -F '.' '{print $4}'`
TIMESTAMP=`date '+%Y_%m_%d_%H_%M'`
OUTPUT=record_${IP_ADDRESS}_${TIMESTAMP}.mjpeg
WIDTH=1500
HEIGHT=1520

rpicam-vid -f --width $WIDTH --height $HEIGHT --codec mjpeg -t $DURATION -o $OUTPUT
