import pandas as pd
import random

data = []

for i in range(500):
    hours_studied = random.randint(1, 10)
    sleep_hours = random.randint(4, 9)
    phone_usage = random.randint(1, 8)

    productivity = (
        hours_studied * 10 +
        sleep_hours * 5 -
        phone_usage * 7 +
        random.randint(-10, 10)
    )

    data.append([hours_studied, sleep_hours, phone_usage, productivity])

df = pd.DataFrame(data, columns=[
    "Hours_Studied",
    "Sleep_Hours",
    "Phone_Usage",
    "Productivity_Score"
])

df.to_csv("data/productivity_data.csv", index=False)

print("Dataset generated successfully!")