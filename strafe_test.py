import time
from gpiozero import Motor

# Setup motors
fr = Motor(5, 6)
fl = Motor(19, 26)
bl = Motor(17, 18)
br = Motor(22, 23)

print("Strafing Right for 3 seconds...")
fl.forward(0.6)
fr.backward(0.6)
bl.backward(0.6)
br.forward(0.6)

time.sleep(3)

# Stop everything
fl.stop(); fr.stop(); bl.stop(); br.stop()
print("Done.")
