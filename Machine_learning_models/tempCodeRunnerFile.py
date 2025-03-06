    # Example plot for Naive Forecast vs Actual
    plt.figure(figsize=(10, 5))
    plt.plot(series_data['ds'], series_data['y'], label='Actual', marker='o')
    plt.plot(series_data['ds'], series_data['Naive'],
             label='Forecast (Naive)', marker='x')
    plt.fill_between(series_data['ds'],
                     series_data['Naive-lo-90'],
                     series_data['Naive-hi-90'],
                     color='gray', alpha=0.2, label='90% CI')
    plt.title(f"Naive Forecast vs Actual for {series_id}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Results_diagrams/NaiveForecast_AT_AL_Q.png')
    plt.show()  # Show the plot after saving

    # Plot WMAPE over time
    plt.figure(figsize=(10, 5))
    plt.plot(wmape_by_date.index, wmape_by_date.values, marker='o')
    plt.xlabel('Forecast Date')
    plt.ylabel('WMAPE')
    plt.title('WMAPE Over Time')
    plt.grid(True)
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Results_diagrams/WMAPE_over_time.png')
    plt.show()

    # Plot for series 'US_ZW_U'
    series_id = 'US_ZW_U'
    series_data = results[results['unique_id'] == series_id]
    plt.figure(figsize=(10, 5))
    plt.plot(series_data['ds'], series_data['y'], label='Actual', marker='o')
    plt.plot(series_data['ds'], series_data['Naive'],
             label='Forecast (Naive)', marker='x')
    plt.fill_between(series_data['ds'],
                     series_data['Naive-lo-90'],
                     series_data['Naive-hi-90'],
                     color='gray', alpha=0.2, label='90% CI')
    plt.title(f"Naive Forecast vs Actual for {series_id}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(
        '/Users/mahfuz/Final_project/Final_repo/Results_diagrams/NaiveForecast_US_ZW_U.png')
    plt.show()
