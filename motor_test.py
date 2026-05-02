import time
from gpiozero import Motor

# Pin mapping based on your notes
# Motor(forward_pin, backward_pin)
front_right = Motor(5, 6)   # Swapped because of your 'backward' note
front_left  = Motor(19, 26)
back_left   = Motor(17, 18)
back_right  = Motor(22, 23)

motors = [front_left, front_right, back_left, back_right]
names = ["Front Left", "Front Right", "Back Left", "Back Right"]

def test_motors():
    print("--- Starting Motor Identity Test ---")
    print("Prop the robot up so wheels spin freely!")
    
    for motor, name in zip(motors, names):
        print(f"Testing: {name} (Pins: {motor.forward_device.pin.number}, {motor.backward_device.pin.number})")
        # Spin forward for 2 seconds
        motor.forward(speed=0.6)
        time.sleep(2)
        motor.stop()
        time.sleep(0.5)

if __name__ == "__main__":
    try:
        test_motors()
        print("--- Test Complete ---")
    except KeyboardInterrupt:
        for m in motors: m.stop()
