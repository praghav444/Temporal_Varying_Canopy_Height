#---------------------------------------------------------------------------------------
# Validate Vegetation Height method at wheat sites
#---------------------------------------------------------------------------------------
import pandas as pd; import numpy as np; import glob; import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "serif"
import seaborn as sns
from tqdm import tqdm
import scipy.optimize as optimize
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from meteo_utils import *
#---------------------------------------------------------------------------------------
base_dir = r"Data"
sites_data_name = ['Site1_data', 'Site2_data']
ht_info_files = ['Wheat_CT_site_1_height', 'Wheat_NT_site_2_height']
fig, axs = plt.subplots(len(sites_data_name), 1, figsize=(16, 2.5 * len(sites_data_name)))
fig.suptitle("Wheat Sites", fontsize=16)
plt.subplots_adjust(hspace=0.4)
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)'] 
for i, site_file in enumerate(sites_data_name):
    height_file = ht_info_files[i]
    wheat_site_info = pd.read_csv(os.path.join(base_dir, f"{height_file}.csv"))
    wheat_site_info.columns = ['Date', 'measured_canopy_height', 'tower_height']
    wheat_site_info['Date'] = pd.to_datetime(wheat_site_info['Date'])
    Zm = 2.5    # Tower Height
    test_df = pd.read_csv(os.path.join(base_dir, f"{site_file}.csv"))
    test_df['DateTime'] = pd.to_datetime(dict(year=test_df['Year'],month=test_df['Month'],day=test_df['Day'],hour=test_df['Hour']))
    USTAR = test_df['ustar'].values
    WS = test_df['U'].values
    H = test_df['H'].values
    Ta = test_df['Ta_Avg'].values
    Pa = test_df['Prss_Avg'].values
    VPD = test_df['VPD_Avg_kPa'].values
    ea = calc_e0(Ta) - VPD
    rho_a = calc_rho(Pa, Ta, ea)
    qa = calc_sp_humidity(Ta, VPD, Pa)
    Cp = 0.24 * 4185.5 * (1 + 0.8 * (0.622 * ea / (Pa - ea))) # Specific heat capacity of air (J kg-1 C-1)
    ObukhovLength = -USTAR ** 3 * rho_a * Cp * (Ta+273.15) * (1 + 0.61*qa) / (0.41 * 9.81 * H)
    test_df['z_L'] = Zm/ObukhovLength
    test_df['VH'] = Zm/(0.66 + 0.1*np.exp(0.41*WS/USTAR))   # Vegetation Height equation (https://hess.copernicus.org/articles/23/925/2019/)
    test_df['VH'] = np.where(test_df['z_L']<0, test_df['VH'], np.nan)
    test_df = test_df.set_index(['DateTime'])
    test_df['Date'] = test_df.index.date
    daily_vh_list = []
    for day, group in test_df.groupby('Date'):
        vh_values = group['VH'].dropna()
        if len(vh_values) < 5:  # Optional: skip days with too few records
            continue
        lower = np.percentile(vh_values, 10)
        upper = np.percentile(vh_values, 90)
        vh_filtered = vh_values[(vh_values >= lower) & (vh_values <= upper)] # Outlier removal
        daily_vh = vh_filtered.mean()
        daily_vh_list.append({'Date': pd.to_datetime(day), 'VH': daily_vh})
    daily_df = pd.DataFrame(daily_vh_list).set_index('Date').sort_index()
    daily_df['VH_smoothed'] = daily_df['VH'].rolling(window=30, min_periods=15, center=True).mean()  # Moving Average of 30 days
    
    merge_df = pd.merge(
        wheat_site_info,
        daily_df[['VH_smoothed']].reset_index(),
        how='left',
        left_on='Date',
        right_on='Date'
    )
    merge_df = merge_df.dropna(subset=['VH_smoothed'])

    rho, _ = spearmanr(merge_df['measured_canopy_height'], merge_df['VH_smoothed'])
    #
    ax = axs[i]
    ax.plot(daily_df.index, daily_df['VH_smoothed'], label='Modeled Canopy Height', color='orange')
    ax.scatter(merge_df['Date'], merge_df['measured_canopy_height'], 
                color='black', label='Observed Canopy Height', zorder=5)
    ax.set_ylabel("Canopy Height (m)", fontsize=16)
    ax.text(0.1, 0.9, f"R: {rho:.2f}", transform=ax.transAxes, fontsize=16)
    ax.set_xlabel("")
    ax.text(0.01, 0.92, subplot_labels[i], transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.legend()
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(False)
plt.tight_layout()
fig.savefig("comparison_plot.png", dpi=600, bbox_inches='tight')
plt.show()
