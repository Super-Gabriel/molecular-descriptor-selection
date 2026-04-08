# molecular-descriptor-selection

Pipeline para calcular descriptores, reducirlos y entrenar un modelo de Random Forest para predecir la actividad biológica de una molécula.

## scores del modelo

### Sin normalizar (con Ipc)

- R^2_train: 0.9510217031562431
- R^2_test: 0.6459124946748234
- RMSE_test: 0.7213500267965754

### Sin normalizar (sin Ipc)

- R^2_train: 0.9478678290516253
- R^2_test: 0.6854262556004221
- RMSE_test: 0.6799108741677797

### normalizado (con Ipc)

- R^2_train: 0.9517588964823063
- R^2_test: 0.6402410421272615
- RMSE_test: 0.7271040429507747

### normalizado (sin Ipc)

- R^2_train: 0.9478431987581064
- R^2_test: 0.6874889281389356
- RMSE_test: 0.6776781067563064

