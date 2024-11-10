#!/usr/bin/env python3
# NeoPixel library strandtest example
# Author: Tony DiCola (tony@tonydicola.com)
#
# Direct port of the Arduino NeoPixel library strandtest example.  Showcases
# various animations on a strip of NeoPixels.

import time
from rpi_ws281x import *
import argparse

# LED strip configuration:
LED_PIN = 18  # GPIO pin connected to the pixels (18 uses PWM!).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10  # DMA channel to use for generating signal (try 10)
LED_INVERT = False  # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0  # set to '1' for GPIOs 13, 19, 41, 45 or 53
LED_TYPE = WS2812_STRIP


# Define functions which animate LEDs in various ways.
def colorWipe(strip, color, wait_ms=1000, led_array=None):
    """Wipe color across display a pixel at a time."""
    if not led_array:
        led_array = [i for i in range(strip.numPixels())]

    for i in led_array:
        strip.setPixelColor(i, color)
        strip.show()
        time.sleep(wait_ms / 1000.0)


def theaterChase(strip, color, wait_ms=50, iterations=10):
    """Movie theater light style chaser animation."""
    for j in range(iterations):
        for q in range(3):
            for i in range(0, strip.numPixels(), 3):
                strip.setPixelColor(i + q, color)
            strip.show()
            time.sleep(wait_ms / 1000.0)
            for i in range(0, strip.numPixels(), 3):
                strip.setPixelColor(i + q, 0)


def wheel(pos):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 85:
        return Color(pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return Color(255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return Color(0, pos * 3, 255 - pos * 3)


def rainbow(strip, wait_ms=20, iterations=1):
    """Draw rainbow that fades across all pixels at once."""
    for j in range(256 * iterations):
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, wheel((i + j) & 255))
        strip.show()
        time.sleep(wait_ms / 1000.0)


def rainbowCycle(strip, wait_ms=20, iterations=5):
    """Draw rainbow that uniformly distributes itself across all pixels."""
    for j in range(256 * iterations):
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, wheel((int(i * 256 / strip.numPixels()) + j) & 255))
        strip.show()
        time.sleep(wait_ms / 1000.0)


def theaterChaseRainbow(strip, wait_ms=50):
    """Rainbow movie theater light style chaser animation."""
    for j in range(256):
        for q in range(3):
            for i in range(0, strip.numPixels(), 3):
                strip.setPixelColor(i + q, wheel((i + j) % 255))
            strip.show()
            time.sleep(wait_ms / 1000.0)
            for i in range(0, strip.numPixels(), 3):
                strip.setPixelColor(i + q, 0)


def strobe_effect(strip, delay_time, led_array=None):
    """Strobing effect function."""
    for idx, color in led_array:
        strip.setPixelColor(idx, color)
    strip.show()
    time.sleep(delay_time)
    for idx, color in led_array:
        strip.setPixelColor(idx, Color(0, 0, 0))
    strip.show()
    time.sleep(delay_time)


def parse_led_config(led_config: str) -> list:
    led_config_list = led_config.split(",")
    idx_and_color = []
    for elem in led_config_list:
        idx, r, g, b = elem.split(":")
        r, g, b = [int(i) for i in [r, g, b]]
        if "-" in idx:
            start, stop = [int(i) for i in idx.split("-")]
            for i in range(start, stop + 1):
                idx_and_color.append([i, Color(r, g, b)])
        else:
            idx_and_color.append([int(idx), Color(r, g, b)])

    return idx_and_color


# Main program logic follows:
if __name__ == "__main__":
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--led-count", type=int, default=30, help="number of LEDs in the strip"
    )
    parser.add_argument(
        "-b", "--brightness", type=int, default=255, help="brightness level 0 - 255"
    )
    parser.add_argument("-t", "--timeout", type=int, default=3, help="seconds to wait")
    parser.add_argument(
        "--color", nargs=3, type=int, default=[255, 255, 255], help="color tuple"
    )
    parser.add_argument(
        "--led-config",
        type=str,
        default=None,
        help=(
            "list of array indexes with color specification. Example: 6-10:255:0:0,11-16:0:255:0 = "
            "this will turn on LED 6-10 with RED, 11-16 with BLUE"
        ),
    )
    parser.add_argument(
        "--strobe",
        type=float,
        default=None,
        help="Do a strobing effect with given time delay. Defaults to None.",
    )
    args = parser.parse_args()

    # Create NeoPixel object with appropriate configuration.
    strip = Adafruit_NeoPixel(
        args.led_count,
        LED_PIN,
        LED_FREQ_HZ,
        LED_DMA,
        LED_INVERT,
        args.brightness,
        LED_CHANNEL,
        strip_type=LED_TYPE,
    )
    # Intialize the library (must be called once before other functions).
    strip.begin()

    color = Color(*args.color)
    timeout = args.timeout
    if not args.led_config:
        led_array = [[i, color] for i in range(args.led_count)]
    else:
        led_array = parse_led_config(args.led_config)

    if args.strobe:
        start_time = time.time()
        while True:
            strobe_effect(
                strip=strip,
                delay_time=args.strobe,
                led_array=led_array,
            )
            curr_time = time.time()
            if (curr_time - start_time) >= timeout:
                break
    else:
        for idx, color in led_array:
            strip.setPixelColor(idx, color)
        strip.show()
        time.sleep(timeout)

    colorWipe(strip, Color(0, 0, 0), 10)
    strip.show()
