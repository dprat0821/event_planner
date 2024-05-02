import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Generate group sizes using a truncated normal distribution approach
def generate_truncated_normal(mean=50, sd=15, low=5, upp=200):
    ret = low - 1
    while ret < low or ret > upp:
        ret = int(np.random.normal(mean, sd))
    return ret

def generated_weighted_choices(common_choices, uncommon_choices, common_weight):
    # Weigh the probability of choosing common vs uncommon choices
    weights = [common_weight, 1 - common_weight]

    if np.random.choice([True, False], p=weights):
        return choice(common_choices)
    else:
        return choice(uncommon_choices)

from re import U
import pandas as pd
import numpy as np
from random import choice, randint, choices


# Number of records as a variable
num_records = 1000

# Predefined locations and months
office_locations = ['Toronto', 'Vancouver', 'San Francisco', 'New York']
event_locations = ['Paris', 'Niagara', 'Cancun']
common_durations = [3, 7, 14]
uncommon_durations = list(range(1, 21))  # Example range from 1 to 20
common_months = [7,8,9]
uncommon_months = list(range(1, 13))
# Generating correlated data
data = {
    "office_location": [choice(office_locations) for _ in range(num_records)],
    "event_month": [generated_weighted_choices(
                        common_choices=common_months,
                        uncommon_choices=uncommon_months,
                        common_weight=0.75
                        ) for _ in range(num_records)],
    "days_duration": [generated_weighted_choices(
                        common_choices=common_durations,
                        uncommon_choices=uncommon_durations,
                        common_weight=0.75
                        ) for _ in range(num_records)],
    "group_size": [int(generate_truncated_normal(mean=50, sd=15, low=5,upp=200)) for _ in range(num_records)],
    "budget_person_day": [int(generate_truncated_normal(mean=250, sd=100, low=200,upp=600)) for _ in range(num_records)],
    "hotel_id": [randint(1, 5) for _ in range(num_records)],
    "event_location": [choice(event_locations) for _ in range(num_records)],
    "nps_label": [0 for _ in range(num_records)]
}

# Calculate total_budget and add it to the data dictionary
data["total_budget"] = [
    data["days_duration"][i] * data["group_size"][i] * data["budget_person_day"][i]
    for i in range(num_records)
]

### DETERMINE `nps_label`

# Adding the 'prefixed_hotel_id' column
data['prefixed_hotel_id'] = [f"{data['event_location'][i]}_{str(data['hotel_id'][i])}" for i in range(num_records)]

high_nps_hotels = {'Paris_1', 'Niagara_3', 'Cancun_2'}
low_nps_hotels = {'Paris_5', 'Niagara_1', 'Cancun_4'}

high_nps_month = [7,8,9]

def estimate_nps(data, i):
  factor_hotel = 0
  if data['prefixed_hotel_id'][i] in high_nps_hotels:
      factor_hotel = int(generate_truncated_normal(mean=4,sd=1,low=3,upp=6))  # Higher average NPS
  elif data['prefixed_hotel_id'][i] in low_nps_hotels:
      factor_hotel = int(generate_truncated_normal(mean=2,sd=1,low=0,upp=4))
  else:
      factor_hotel = int(generate_truncated_normal(mean=3,sd=1.5,low=2,upp=5))

  factor_month = 0
  if data['event_month'][i] in high_nps_month:
    factor_month = int(generate_truncated_normal(mean=3,sd=1,low=1,upp=4))

  factor_budget = int(generate_truncated_normal(mean=data['budget_person_day'][i]/100,sd=1,low=1,upp=4))

  return max(1, min(10, int(factor_hotel+ factor_month+ factor_budget)))

for i in range(num_records):
    data['nps_label'][i]= estimate_nps(data,i)



### GENERATE DATA FRAME
df = pd.DataFrame(data)


# Save DataFrame to a CSV file
df.to_csv('updated_data.csv', index=False)

# Display the first few rows of the dataframe
print(df.head())
