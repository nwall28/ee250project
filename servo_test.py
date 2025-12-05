import RPi.GPIO as GPIO
import time

SERVO_PIN = 18 #GPIO pin connected to the servo

def setup_servo():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50) #50Hz frequency
    pwm.start(7.5) #neutral position
    return pwm

def slap_motion(pwm):
    """Function to perform slap motion with the servo"""
    pwm.ChangeDutyCycle(2.5) #left
    time.sleep(0.2)
    pwm.ChangeDutyCycle(7.5) #neutral
    time.sleep(0.2)
    pwm.ChangeDutyCycle(12.5) #right
    time.sleep(0.2)
    pwm.ChangeDutyCycle(7.5) #neutral


def main():
    pwm = setup_servo()
    try:
        slap_motion(pwm)
    finally:
        pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()  