from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
import pandas as pd
import dill


sched = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv('model/data/30.6 homework.csv')
with open('model/price_cars.pkl', 'rb') as file:
    model = dill.load(file)

@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(5)
    data['preds'] = model['model'].predict(data)
    print(data[['id', 'preds']])



if __name__== '__main__':
    sched.start()