
import os
def mkdir():
    if os.path.exists("./Log")!=True:
        os.mkdir("./Log")
    if os.path.exists("./Saved")!=True:
        os.mkdir("./Saved")
