import  schedule
import time
from date import*

schedule.every(1).minutes.do(processing)
while 1:
    schedule.run_pending()
    time.sleep(1)
