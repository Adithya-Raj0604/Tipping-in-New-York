import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def preprocessing():
    # preprocessing for service type column
    service_mapping = {
        'HV0002': 'juno',
        'HV0003': 'uber',
        'HV0004': 'via',
        'HV0005': 'lyft'
    }

    df['service_name'] = df['hvfhs_license_num'].map(service_mapping)

    one_hot_encoding = pd.get_dummies(df['service_name'], prefix='is')
    df = pd.concat([df, one_hot_encoding], axis=1)

    # preprocess tip amount
    df = df.loc[df['base_passenger_fare'] > 0]
    df['tip_percent'] = df['tips'] / (df['base_passenger_fare'] + df['tolls'] + df['sales_tax'])

    df['shared_with_friend'] = (df['shared_request_flag'] == "Y") & (df['shared_match_flag'] == "N")
    df['shared_with_stranger'] = df['shared_match_flag'] == "Y"


def correlation_analysis():
    valid_columns_for_corr = [
        'tips', 
        'tip_percent', 
        'is_juno', 
        'is_uber', 
        'is_via', 
        'is_lyft', 
        'temperature_2m (°C)', 
        'rain (mm)', 
        'windspeed_10m (km/h)', 
        'precipitation (mm)',
        'shared_with_friend',
        'shared_with_stranger'
    ]
    correlation_matrix = df[valid_columns_for_corr].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 8})
    plt.show()


def statistical_tests():
    # H0: there is no significant difference between the tip amount of shared rides or single rides
    # H1: there is a signficant difference


    # RQ1
    with_ride_share = df.loc[df['shared_request_flag'] == "Y"]
    no_ride_share = df.loc[df['shared_request_flag'] == "N"]

    t_stat, p_value = stats.mannwhitneyu(with_ride_share['tip_percent'], no_ride_share['tip_percent'], alternative='two-sided')

    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference.")


    # RQ2
    juno_df = df.loc[df['service_name'] == "juno"]
    uber_df = df.loc[df['service_name'] == "uber"]
    via_df = df.loc[df['service_name'] == "via"]
    lyft_df = df.loc[df['service_name'] == "lyft"]

    stat, p_value = stats.kruskal(juno_df['tip_percent'], uber_df['tip_percent'], via_df['tip_percent'], lyft_df['tip_percent'])

    print(f"Kruskal-Wallis H-statistic: {stat}")
    print(f"P-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference.")


    # RQ3
    rained_df = df.loc[df['rain (mm)'] > 0]
    no_rain_df = df.loc[df['rain (mm)'] <= 0]

    t_stat, p_value = stats.mannwhitneyu(rained_df['tip_percent'], no_rain_df['tip_percent'], alternative='two-sided')

    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference.")
    


def main():
    preprocessing()
    correlation_analysis()
    statistical_tests()

df = pd.read_csv('sampled_taxi_weather_data.csv')
main()

