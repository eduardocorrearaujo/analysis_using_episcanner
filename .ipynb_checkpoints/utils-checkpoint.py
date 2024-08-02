import numpy as np 
import pandas as pd
import seaborn as sns
import geopandas as gpd 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b\n %Y')

MEDIUM_SIZE = 12
LARGE_SIZE = 14

predictors = [
'year',
'casos_01',
'casos_1_3',
'casos_1_4',
'populacao_1',
'peak_week_1',
'R0_1',
't_end_1', 
'ep_dur_1',
'dummy_ep',      
'temp_med_4',
'temp_amp_4',
'temp_max_4',
'temp_min_4', 
'umid_min_4',
'umid_max_4', 'umid_amp_4', 'enso_4',
'precip_tot_4', 'rainy_day_4', 'thr_temp_min_4', 'thr_temp_amp_4', 'thr_umid_med_4',      
'temp_med_1_current',
'temp_amp_1_current',
'temp_max_1_current',
'temp_min_1_current',
'precip_tot_1_current',
'rainy_day_1_current',
'enso_1_current', 
'latitude', 'longitude']

def plot_hist(ax, df, region):
  
    ax.hist(df.loc[df.year < 2023].peak_week, alpha = 0.7, label = 'Before 2023')
    
    ax.hist(df.loc[df.year == 2023].peak_week, alpha = 0.7, label = '2023')
    
    ax.legend()
    
    ax.set_title(f'Peak week - {region}')


def plot_preds(ax, dft, hue = None, bounds = None):
    '''
    Function to create a scatter plot of the predictions given a dataset with target and pred columns 

    Parameters
    ----------
    ax: matplotlib ax 

    dft: pd.DataFrame 
        A DataFrame with the columns pred and target. 
    
    Returns
    -------
    None 
    '''

    sns.scatterplot(data=dft, x="pred", y="target", hue=hue, ax = ax)
    ax.plot(bounds, bounds, ls = '--', color = 'black')

    ax.set_xlim([bounds[0], bounds[-1]] )

    ax.set_ylim([bounds[0], bounds[-1]]) 

    ax.set_xlabel('Predicted Values')

    ax.set_ylabel('Observed Values')

    ax.grid()
    if hue != None: 
        ax.legend(bbox_to_anchor=(0.75,0.7))


def get_train_test(df,  target = 'ep_pw', predictors = predictors, test_year = 2019):
    '''
    Function to get the train and test samples to apply the regression model 
    '''
    df_train = df.loc[df.year < test_year]
    df_test = df.loc[df.year == test_year]
    
    X_train = df_train[predictors]
    y_train = df_train[target]#.values.reshape(-1,1)
    X_test = df_test[predictors]
    y_test = df_test[target]#.values.reshape(-1,1)
    
    return X_train, y_train, X_test, y_test

def get_train_for(df,  predictors = predictors):
    '''
    Function to get the train and test samples to apply the regression model 
    '''
    
    X = df[predictors]
    
    return X
    

@np.vectorize
def richards(L,a,b,t,tj):
    '''
    Function that applies the richard's model 
    '''
    j=L-L*(1+a*np.exp(b*(t-tj)))**(-1/a)
    return j



def get_data_pars(state, agravo): 
    '''
    Function to load the episcanner parameters saved
    '''
    df_state = pd.read_parquet(f'./data/{state}_{agravo}.parquet')
    pars = pd.read_csv(f'./params/pars_{state}_{agravo}_nov.csv.gz', index_col = 'Unnamed: 0')

    return df_state, pars

def get_richards_pars(sir):
    """Returns the Richards parameters based on the estimated SIR parameters 
    """
    
    pars = {
        'b': sir['beta'] - sir['gamma'],
        'a': sir['alpha'], 
        'L': sir['total_cases'],
        'tj': sir['peak_week'], 
        't': np.arange(0,52)
        
    }
    return pars


def plot_curve(ax, df_state,  pars):
    '''
    Function to plot the estimated richard curve by year 
    ''' 
    
    df_state = df_state.sort_index()
    
    year = pars['year']
    
    df_state = df_state.loc[(df_state.municipio_geocodigo == pars['geocode']) & ((df_state.index.year == year-1)
                           | (df_state.index.year == year))].iloc[44: 44+52]

    df_state['casos_cum'] = df_state.casos.cumsum()
    
    p = get_richards_pars(pars)
        
    curve = richards(**p)    
    
    df_state.index = pd.to_datetime(df_state.index)
    
    ax.fill_between(df_state.index, df_state['casos_cum'], alpha=0.3, color='r', label=f'data_{year}')
    ax.plot(df_state.index, curve,label='model')
    ax.axvline(df_state.iloc[round(pars['peak_week'])].name, color = 'red', ls = '--' ,label = 'peak_week')
    
    ax.plot(df_state.index, np.abs(curve - df_state.casos_cum), marker='+', label='Abs. residuals')
    ax.legend()
    ax.xaxis.set_major_formatter(myFmt)
    

    
    return
            

gdf_reg = gpd.read_file('./data/shapefile_region.gpkg')
gdf_reg['centroid'] = gdf_reg.centroid

gdf_reg['name_region'] = gdf_reg['region'].replace({'North': 'N', 'North East':'NE',
                                                    'Southeast':'SE', 'South':'S', 'Midwest': 'MW'})

gdf_reg.head()

def plot_dengue_vs_chik(dfd, dfc, maps, column, title, title_colorbar, save = False, year = None): 
    '''
    Function to plot the maps of dengue and chikungunya side by side 
    '''
    fig, ax = plt.subplots(1,2, figsize = (11, 7))

    gdf_reg.boundary.plot(color = 'black', ax =ax[0], linewidth=0.7)

    gdf_reg.boundary.plot(color = 'black', ax=ax[1], linewidth=0.7)
    

    min_v = min(dfd[column].min(), dfc[column].min())

    max_v = max(dfd[column].max(), dfc[column].max())
    
    maps.plot(ax=ax[0], color = 'lightgray')

    dfd.plot(column = column,legend = False, ax = ax[0], cmap = 'RdYlGn_r', vmin= min_v, vmax = max_v, legend_kwds={
            "shrink":.5
        })

    ax[0].set_axis_off()

    ax[0].set_title(f'Dengue')
    
    maps.plot(ax=ax[1], color = 'lightgray')
    
    dfc.plot(column = column,legend = False, ax = ax[1], cmap = 'RdYlGn_r', vmin= min_v, vmax = max_v, legend_kwds={
            "shrink":.5
        })
    
    
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=min_v, vmax = max_v))

    cax = fig.add_axes([0.87, 0.3, 0.015, 0.35])

    cbar = fig.colorbar(sm, cax= cax)
    
    cbar.set_label(label = title_colorbar, rotation = 270, labelpad = 15)
    
    ax[1].set_axis_off()

    ax[1].set_title(f'Chikungunya')
    
    fig.suptitle(title, fontsize = 14, y = 0.92)

    l, b, h, w = .275, .25, .25, .5
    
    ax2 = fig.add_axes([l, b, w, h])

    gdf_reg.boundary.plot(color = 'gray', ax =ax2, linewidth=0.7)
    gdf_reg.plot(color='lightgray', ax =ax2)

    ax2.set_axis_off()

    for idx, row in gdf_reg.iterrows():
        ax2.text(row['centroid'].x, row['centroid'].y, row['name_region'], 
            ha='center', fontsize=10)
    
    plt.subplots_adjust(wspace = -0.05)
    
    if save: 

        if year is None: 
        
            plt.savefig(f'./figures/map_{column}.png', bbox_inches='tight',  dpi = 300)

        else: 
            plt.savefig(f'./figures/map_{column}_{year}.png', bbox_inches='tight',  dpi = 300)
 
    plt.show()
    

def scatter_ep_dur_R0(pars_end, agravo = 'dengue'):
    '''
    Plot duration vs reproduction number by disease 
    '''
 
    sns.catplot(
        data=pars_end.loc[(pars_end.year > 2010) ], x="ep_dur", y="R0", hue="region",
        native_scale=True, zorder=1, 
        height=5.0, aspect=8/5,
        kind='strip'
    )

    ax = plt.gca()

    ax.set_ylabel(r'${\cal R}_0$', fontsize = MEDIUM_SIZE)

    ax.set_xlabel('Epidemic duration (weeks)', fontsize = MEDIUM_SIZE)

    plt.savefig(f'./figures/duration_{agravo}_R0_BR.png', bbox_inches='tight', dpi = 300)

    plt.show()
