import time
from gpiozero import Motor

# Setup
fr = Motor(5, 6)
fl = Motor(19, 26)
bl = Motor(17, 18)
br = Motor(22, 23)

def stop_all():
    fr.stop(); fl.stop(); bl.stop(); br.stop()

print("Testing STRAFE RIGHT...")
# For a proper X-pattern to move right:
fl.backward(0.6)
fr.forward(0.6)
bl.forward(0.6)
br.backward(0.6)

time.sleep(2)
stop_all()
print("Test complete.")
