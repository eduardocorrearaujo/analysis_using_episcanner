# Analysis using Episcanner parameters

The episcanner parameters are a set of epidemiological parameters, such as Reproduction Number, peak week, begin week, end week, and duration, computed for all the dengue and chikungunya epidemics notified in Brazil. The methodology used to compute these parameters are explained here: [Large-scale Epidemiological modeling: Scanning for Mosquito-Borne Diseases Spatio-temporal Patterns in Brazil](https://arxiv.org/abs/2407.21286) 

The episcanner parameters saved in the `data` folder can be downloaded using the example [here](https://api.mosqlimate.org/docs/datastore/GET/episcanner/). The name of the parameters file contains the disease and state names that they refer to. There is also an online dashboard with the parameters estimated by municipality available [here](https://info.dengue.mat.br/epi-scanner/). 


The table of features used in the HGBR model to predict the peak week (`train_HGBR_model.ipynb` notebook) is presented in the table below: 

| Feature description                                                                                                                   | Type            |
|---------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| year:  year of the peak week that are being predicted.                                                                               | temporal        |
| casos\_01:   sum of cases in the January of the year whose peak week are being predicted.                                             | epidemiological |
| casos\_1\_3: sum of cases on the third quarter of the previous year.                                                                  | epidemiological |
| casos\_1\_4: sum of cases on the fourth quarter of the previous year.                                                                 | epidemiological |
| populacao\_1: population in the previous year.                                                                                        | demographic     |
| peak\_week\_1:  peak week estimated on the previous year.                                                                             | epidemiological |
| R0\_1: reproduction number estimated on the previous year.                                                                            | epidemiological |
| ep\_dur\_1: epidemic duration estimated on the previous year.                                                                         | epidemiological |
| dummy\_ep: 1 if the previous year were an epidemic identified by Episcanner and 0 otherwise.                                          | epidemiological |
| temp\_med\_4: average of the average temperature over the fourth quarter of the previous year.                                        | climatic        |
| temp\_amp\_4: average of the temperature amplitude  over the fourth quarter of the previous year.                                     | climatic        |
| temp\_max\_4: average of the maximum temperature over the fourth quarter of the previous year.                                        | climatic        |
| temp\_min\_4: average of the minimum temperature over the fourth quarter of the previous year.                                        | climatic        |
| umid\_min\_4: average of the minimum humidity over the fourth quarter of the previous year.                                           | climatic        |
| umid\_max\_4: average of the maximum humidity over the fourth quarter of the previous year.                                           | climatic        |
| umid\_amp\_4:average of the  humidity amplitude over the fourth quarter of the previous year.                                         | climatic        |
| enso\_4: average of the  multivariate ENSO (El Niño-Southern Oscillation) in the fourth quarter of the previous year.                 | climatic        |
| precip\_tot\_4: sum of the total precipitation over the fourth quarter of the previous year.                                          | climatic        |
| rainy\_day\_4: sum of days with rain (precipitation above zero) over the fourth quarter of the previous year.                         | climatic        |
| thr\_temp\_min\_4: sum of days with minimum temperature below 15-celsius degrees                                                      | climatic        |
| thr\_temp\_amp\_4: sum of days with temperature amplitude above celsius degrees.                                                      | climatic        |
| thr\_umid\_med\_4: sum of days with average humidity above 0.8.                                                                       | climatic        |
| temp\_med\_1\_current:  average of the average temperature over the january of the year whose peak week is being predicted.           | climatic        |
| temp\_amp\_1\_current: average of the temperature amplitude over the january of the year whose peak week is being predicted.          | climatic        |
| temp\_max\_1\_current: average of the maximum temperature over the january of the year whose peak week is being predicted.            | climatic        |
| temp\_min\_1\_current: average of the minimum temperature over the january of the year whose peak week is being predicted.            | climatic        |
| precip\_tot\_1\_current: sum of total precipitation over the january of the year whose peak week is being predicted.                  | climatic        |
| rainy\_day\_1\_current:  sum of days with precipitation over the january of the year whose peak week is being predicted.              | climatic        |
| enso\_1\_current: average of the ENSO (El Niño-Southern Oscillation) over the January of the year whose peak week is being predicted. | climatic        |
| latitude of  the city center.                                                                                                         | spatial         |
| longitude of the city center.                                                                                                         | spatial         |


