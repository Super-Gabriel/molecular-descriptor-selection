import os

def build_args(project_path:str):
    raw_dataset_path = project_path + "/datasets/cruzain_dataset.csv"
    descriptors_dataset_path = project_path + "/datasets/cruzain_descriptors.csv"
    reduced_descriptors_dataset_path = project_path + "/datasets/cruzain_reduced_descriptors.csv"

    target_column = "pIC50"
    smiles_column = "SMILES"
    return(
        project_path,
        raw_dataset_path,
        descriptors_dataset_path,
        reduced_descriptors_dataset_path,
        target_column,
        smiles_column
    )

if __name__ == "__main__":
    from src.Driver import Driver
    project_path = os.getcwd()
    args = build_args(project_path)
    driver = Driver(args)
    driver.run(normalize_features=False)
    