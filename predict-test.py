import requests

url = 'http://student-success-serving.eba-yzdssnnc.eu-west-2.elasticbeanstalk.com/predict'

student = {
    'marital_status': 1.0,
    'application_mode': 6.0,
    'application_order': 1.0,
    'course': 11.0,
    'daytime/evening_attendance': 1.0,
    'previous_qualification': 1.0,
    'nationality': 1.0,
    "mother's_qualification": 1.0,
    "father's_qualification": 3.0,
    "mother's_occupation": 4.0,
    "father's_occupation": 4.0,
    'displaced': 1.0,
    'educational_special_needs': 0.0,
    'debtor': 0.0,
    'tuition_fees_up_to_date': 0.0,
    'gender': 1.0,
    'scholarship_holder': 0.0,
    'age_at_enrollment': 19.0,
    'international': 0.0,
    'curricular_units_1st_sem_(credited)': 0.0,
    'curricular_units_1st_sem_(enrolled)': 6.0,
    'curricular_units_1st_sem_(evaluations)': 6.0,
    'curricular_units_1st_sem_(approved)': 6.0,
    'curricular_units_1st_sem_(grade)': 14.0,
    'curricular_units_1st_sem_(without_evaluations)': 0.0,
    'curricular_units_2nd_sem_(credited)': 0.0,
    'curricular_units_2nd_sem_(enrolled)': 6.0,
    'curricular_units_2nd_sem_(evaluations)': 6.0,
    'curricular_units_2nd_sem_(approved)': 6.0,
    'curricular_units_2nd_sem_(grade)': 13.666666666666666,
    'curricular_units_2nd_sem_(without_evaluations)': 0.0,
    'unemployment_rate': 13.9,
    'inflation_rate': -0.3,
    'gdp': 0.79
 }
response = requests.post(url, json=student).json()
print(); print(response)