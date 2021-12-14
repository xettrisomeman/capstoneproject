import requests
import random
import regex as re

from typing import List


port = 5000

url = f"http://localhost:{port}/predict/"


class Data:
    # got from train data min age and max age
    age = random.randint(17, 98)
    job = random.choices(['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                          'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = random.choices(
        ['divorced', 'married', 'single', 'unknown'])
    education = random.choices(['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                'illiterate', 'professional.course', 'university.degree', 'unknown'])
    default = random.choices(['yes', 'no', 'unknown'])
    housing = random.choices(['no', 'yes', 'unknown'])
    loan = random.choices(['no', 'yes', 'unknown'])
    contact = random.choices(['cellular', 'telephone'])
    month = random.choices(
        ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = random.choices(['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = random.randint(0, 4199)
    campaign = random.randint(1, 56)
    pdays = random.randint(-1, 27)
    previous = random.randint(0, 7)
    poutcome = random.choices(['failure', 'nonexistent', 'success'])
    emp_var_rate = random.uniform(-3.4, 1.4)
    cons_price_idx = random.uniform(92.2, 94.7)
    cons_conf_idx = random.uniform(-50.8, -27.9)
    euribor3m = random.uniform(0.6, 5.0)
    nr_employed = random.randint(4963, 5228)

    @classmethod
    def get_data_random(cls):
        data = {}
        for keys, value in cls.__dict__.items():
            if not keys.startswith("__") and keys != "get_data_random":
                if isinstance(value, int) or isinstance(value, float):
                    data[keys] = value
                else:
                    data[keys] = "".join(value)
        return data


data = Data()
prediction = requests.post(url, json=data.get_data_random()).json()

print(prediction)
