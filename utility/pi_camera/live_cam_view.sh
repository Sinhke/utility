rpicam-vid --width 2312 --height 1786 --codec mjpeg -t 0 -f --autofocus-mode manual --lens-position 10.0 


raspivid -o video.h264 -t 10000 -b 25000000


sudo apt-get install gpac


# build gpac from source
sudo apt install build-essential pkg-config g++ git cmake yasm zlib1g-dev
git clone https://github.com/gpac/gpac.git
cd gpacc

./configure --static-bin
make

sudo make install
