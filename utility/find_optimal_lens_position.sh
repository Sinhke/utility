
for dist in  7 7.5 8 9 9.5 10 11 12 13 14 15
do
	echo "DIST=" $dist 
	rpicam-vid --width 2312 --height 1786 --codec mjpeg -t 5000 -f --autofocus-mode manual --lens-position $dist -o temp/lens_pos_$dist.mjpeg
done
