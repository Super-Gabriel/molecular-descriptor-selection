import pandas as pd
from .descriptors_generator import generate_descriptors
from .mRMR import mRMR_feature_selection

class Driver:
    def __init__(self, args:tuple):
        (
            self.project_path,
            self.raw_dataset_path,
            self.descriptors_dataset_path,
            self.reduced_descriptors_dataset_path,
            self.target_column,
            self.smiles_column
        ) = args

        


    def run(self):
        raw_df = pd.read_csv(self.raw_dataset_path)
        descriptors_df = self.generate_descriptors_dataset(raw_df)
        reduced_descriptors_df = self.reduce_descriptors_dataset(descriptors_df)

    def reduce_descriptors_dataset(self, descriptors_df:pd.DataFrame):
        print("\n\tReduciendo descriptores con mRMR...")
        reduced_descriptors = mRMR_feature_selection(descriptors_df, self.target_column) # aplicando mRMR
        reduced_descriptors_df = descriptors_df[[self.target_column] + reduced_descriptors]
        reduced_descriptors_df.to_csv(self.reduced_descriptors_dataset_path, index=False)

        return reduced_descriptors_df

    def generate_descriptors_dataset(self, raw_df:pd.DataFrame):
        print("\n\tGenerando descriptores...")
        descriptors = raw_df[self.smiles_column].apply(generate_descriptors)
        descriptors = pd.DataFrame(descriptors.tolist(), index=raw_df.index)
        descriptors = descriptors.dropna(axis=0, how='all') # eliminando fila de moleculas con fallo de TODOS los descriptores
        descriptors = descriptors.dropna(axis=1, how='any') # eliminando columnas de descriptores que generaron algun NaN
        descriptors = pd.concat([raw_df[[self.target_column]], descriptors], axis=1, join='inner') # 'inner' para conservar solamente las filas/moléculas que sobrevivieron
        descriptors.to_csv(self.descriptors_dataset_path, index=False)

        return descriptors