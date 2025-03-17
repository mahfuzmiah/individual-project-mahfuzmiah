
    composite_variability = {}

    for uid, group in df.groupby('unique_id'):
        group = group.set_index('ds').sort_index()
        try:
            # Perform seasonal decomposition (adjust period as needed; here period=4 for quarterly data)
            result = seasonal_decompose(group['y'], model='additive', period=4)

            # Compute the standard deviation of the trend and residual components
            trend_std = np.nanstd(result.trend)
            resid_std = np.nanstd(result.resid)

            # Create a composite metric (you can adjust weights if desired)
            composite_score = trend_std + resid_std
            composite_variability[uid] = composite_score
        except Exception as e:
            print(f"Could not decompose series {uid}: {e}")

    # Find the series with the highest composite variability
    if composite_variability:
        best_uid = max(composite_variability, key=composite_variability.get)
        print("Series with highest composite (trend + residual) variability:", best_uid)

        # Decompose the "most interesting" series again and plot its components
        best_series = df[df['unique_id'] ==
                         best_uid].set_index('ds').sort_index()
        result = seasonal_decompose(
            best_series['y'], model='additive', period=4)
        fig = result.plot()
        fig.suptitle(f'Seasonal Decomposition for {best_uid}', y=1.02)
        plt.show()
    else:
        print("No series could be decomposed.")