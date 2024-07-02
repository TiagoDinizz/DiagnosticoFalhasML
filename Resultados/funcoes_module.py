""" 
Module to define statistical metrics.

This module contains functions so can be used in the main code

Functions:
----------

"""

import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

""" 
Module to define statistical metrics.

This module contains functions to calculate various statistical metrics for a given input vector.

Functions:
----------
rms(input_vector: np.ndarray) -> float:
    Calculates the root mean square of the input vector.

pk_pk(input_vector: np.ndarray) -> float:
    Calculates the peak-to-peak value of the input vector.

kurtosis(input_vector: np.ndarray) -> float:
    Calculates the kurtosis of the input vector.

crest_factor(input_vector: np.ndarray) -> float:
    Calculates the crest factor of the input vector.

crest_factor_plus(input_vector: np.ndarray) -> float:
    Calculates the crest factor plus of the input vector.

skewness(input_vector: np.array) -> float:
    Calculates the skewness of a given input vector.

shape_factor(input_vector: np.ndarray) -> float:
    Calculates the shape factor of the input vector.

clearance_factor(input_vector: np.ndarray) -> float:
    Calculates the clearance factor of the input vector.

impulse_factor(input_vector: np.ndarray) -> float:
    Calculates the impulse factor of the input vector.
"""

import numpy as np
from scipy import stats


def mean_value(input_vector: np.ndarray) -> float:
    input_vector = np.asarray(input_vector)
    mean = np.mean(input_vector)

    return mean


def std_value(input_vector: np.ndarray) -> float:
    input_vector = np.asarray(input_vector)
    std = np.std(input_vector)

    return std


def min_value(input_vector: np.ndarray) -> float:
    input_vector = np.asarray(input_vector)
    min = np.min(input_vector)

    return min


def max_value(input_vector: np.ndarray) -> float:
    input_vector = np.asarray(input_vector)
    max = np.max(input_vector)

    return max


def rms(input_vector: np.ndarray) -> float:
    """
    Calculates the root mean square of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the root mean square is to be calculated.

    Returns:
    --------
    float:
        Root mean square of the input vector.
    """
    input_vector = np.asarray(input_vector)
    input_squared = input_vector**2
    mean = np.mean(input_squared)
    rms = np.sqrt(mean)

    return rms


def pk_pk(input_vector: np.ndarray) -> float:
    """
    Calculates the peak-to-peak value of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the peak-to-peak value is to be calculated.

    Returns:
    --------
    float:
        Peak-to-peak value of the input vector.
    """
    input_vector = np.asarray(input_vector)
    pkpk = input_vector.max() - input_vector.min()

    return pkpk


def kurtosis(input_vector: np.ndarray) -> float:
    """
    Calculates the kurtosis of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the kurtosis is to be calculated.

    Returns:
    --------
    float:
        Kurtosis of the input vector.
    """
    input_vector = np.asarray(input_vector)
    kurt = stats.kurtosis(input_vector, fisher=False)
    return kurt if not np.isnan(kurt) else 0


def crest_factor(input_vector: np.ndarray) -> float:
    """
    Calculates the crest factor of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the crest factor is to be calculated.

    Returns:
    --------
    float:
        Crest factor of the input vector.
    """
    input_vector = np.asarray(input_vector)
    my_pkvalue = np.max(np.abs(input_vector))
    my_rms = rms(input_vector)
    cf = my_pkvalue / my_rms if my_rms != 0 else 0.0

    return cf


"""
def crest_factor_plus(input_vector: np.ndarray) -> float:
#    
    Calculates the crest factor plus of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the crest factor plus is to be calculated.

    Returns:
    --------
    float:
        Crest factor plus of the input vector.
 #  
    input_vector = np.asarray(input_vector)
    my_rms = rms(input_vector)
    my_pkvalue = np.max(np.abs(input_vector))
    my_cf = crest_factor(input_vector)
    cf_plus = (3 * my_rms + 3 * my_pkvalue + 4 * my_cf) / 10

    return cf_plus
"""


def skewness(input_vector: np.array) -> float:
    """
    Calculates the skewness of a given input vector.

    Parameters:
    -----------
    input_vector (np.array): The input vector for which skewness is to be calculated.

    Returns:
    --------
    float: The skewness of the input vector.
    """
    input_vector = np.asarray(input_vector)
    skew = stats.skew(input_vector)
    return skew if not np.isnan(skew) else 0


def shape_factor(input_vector: np.ndarray) -> float:
    """
    Calculates the shape factor of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the shape factor is to be calculated.

    Returns:
    --------
    float:
        Shape factor of the input vector.
    """
    input_vector = np.asarray(input_vector)
    my_rms = rms(input_vector)
    abs_value_mean = np.mean(np.abs(input_vector))
    sf = my_rms / abs_value_mean if abs_value_mean != 0 else 0.0

    return sf


## 1. Funções para extrair dados a partir do nome do dataset
###################################------------------------------------------######################################################
###################################------------------------------------------######################################################
###################################-----------------PARTE 2------------------######################################################
###################################------------------------------------------######################################################
###################################------------------------------------------######################################################


#### 1.
def extract_info_from_filename(filename, fault):  # "health" , "geral", "miss_teeth"
    if fault == "geral":
        pattern2 = r"^(.*)_(L|M|H)_(torque_circulation)_(\d+rpm|rpm\d+)_(\d+Nm)\.csv$"
        match = re.match(pattern2, filename)
        if match:
            return {
                "Fault": match.group(1),
                "Degree": match.group(2),
                "Categoria": match.group(3),
                "Torque": match.group(5),
                "Rotação": match.group(4),
            }
        return {}

    if fault == "health":
        pattern2 = r"^(.*)_(torque_circulation)_(\d+rpm|rpm\d+)_(\d+Nm)\.csv$"  # essa ordem so funciona se vier torque primeiro (speed circulation)
        match = re.match(pattern2, filename)
        if match:
            return {
                "Fault": match.group(1),
                "Categoria": match.group(2),
                "Torque": match.group(4),
                "Rotação": match.group(3),
            }
        return {}

    if fault == "miss_teeth":
        pattern2 = r"^(.*)_(torque_circulation)_(\d+rpm|rpm\d+)_(\d+Nm)\.csv$"  # essa ordem so funciona se vier torque primeiro (speed circulation)
        match = re.match(pattern2, filename)
        if match:
            return {
                "Fault": match.group(1),
                "Categoria": match.group(2),
                "Torque": match.group(4),
                "Rotação": match.group(3),
            }
        return {}


### 1.4 Adicação da coluna time_index e time


def extract_and_format_dataframe(
    file, file_name, falha
):  # falha = "geral", "miss_teeth", "health"

    df = pd.read_csv(file + file_name)
    info = extract_info_from_filename(file_name, falha)

    for col, value in info.items():
        df[col] = value

    # Organização das colunas
    df.drop(
        columns=[
            "Categoria",
            "speed",
            "motor_vibration_x",
            "motor_vibration_y",
            "motor_vibration_z",
        ],
        inplace=True,
    )
    df.rename(
        columns={"torque": "Input Torque", "Torque": "Output Torque"}, inplace=True
    )

    # Adição do tempo
    df["time_index"] = df.index
    df["time"] = df["time_index"] / 12800
    return df


### 1.5
def format_dataframe_basona(df):  # falha = "geral", "miss_teeth", "health"
    # Organização das colunas
    df.drop(
        columns=[
            "Categoria",
            "speed",
            "motor_vibration_x",
            "motor_vibration_y",
            "motor_vibration_z",
        ],
        inplace=True,
    )
    df.rename(
        columns={"torque": "Input Torque", "Torque": "Output Torque"}, inplace=True
    )

    # Adição do tempo
    df["time_index"] = df.index
    df["time"] = df["time_index"] / 12800
    return df


## 2. Filtragem do intervalo de tempo


def format_time(dataframe, intervalo):  # timeframe = "inicial" , "final", "total"
    df_1 = dataframe
    df_2 = dataframe
    df_1.loc[df_1.time <= 19, "adj_time"] = df_1.time - 11
    df_2.loc[((df_2.time <= 49) & (df_2.time > 41)), "adj_time"] = df_1.time - 33

    df_1 = df_1[(df_1["time"] >= 11) & (df_1["time"] <= 19)]
    df_2 = df_2[(df_2["time"] > 41) & (df_2["time"] <= 49)]
    df3 = pd.concat([df_1, df_2], ignore_index=True)

    if intervalo == "inicial":
        return df_1
    if intervalo == "final":
        return df_2
    if intervalo == "total":
        return df3


## 3. Funções Para a extração de Features


## 3.1 Criação do Dataframe de Features
def features_extract(Dataframe):
    target_columns = Dataframe[
        [
            "Input Torque",
            "gearbox_vibration_x",
            "gearbox_vibration_y",
            "gearbox_vibration_z",
        ]
    ]
    stats_func = {
        "rms": rms,
        "pk_pk": pk_pk,
        "kurtosis": kurtosis,
        "crest_factor": crest_factor,
        "skewness": skewness,
        "shape_factor": shape_factor,
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
    }
    features = {}

    for (
        col_name
    ) in (
        target_columns
    ):  # ok estou aplicando os nomes das colunas nas keys do dicionario
        # quero fazer primeira variável e depois os 9 statistics dessa variiável
        for nome, func in stats_func.items():  # para cada nome da função e função

            features[f"{col_name}.{nome}"] = func(
                Dataframe[col_name]
            )  # printar a variável, nome da função e aplicar a função no resultado
            # tudo isso sendo salvo no dicionário {features}
    New_Dataframe = pd.DataFrame([features], index=[0])

    # features.items()
    return New_Dataframe


### 3.2 Funções para splitar em time frames o dataset de Features
def features_timeframe(df_feature, timeframe):  # timeframe = 0.1 ; 0.2 ; 0.5 ; 1s
    time_interval = timeframe  # 0.1 second
    data = df_feature
    # Create an empty DataFrame to store the features
    features_df = pd.DataFrame()

    # Determine the starting and ending times for each segment
    start_time = data["adj_time"].min()
    end_time = data["adj_time"].max()
    index = []
    # ok - básico #

    # Iterate over each time segment
    current_time = start_time
    while current_time + time_interval <= end_time:
        # Select the data for the current second
        mask = (data["adj_time"] >= current_time) & (
            data["adj_time"] < current_time + time_interval
        )  # arquivar o intervalo desejado na variável
        df_second = data.loc[mask]  # filtrar o invervalo desejado no dataframe

        # Calculate features if the slice has data          ## Aqui calcula o que quiser no intervalo
        if (
            not df_second.empty
        ):  ## essa parte do código que vou modificar e colocar oq quero
            current_features_df = features_extract(df_second)
            index.append(current_time + time_interval)

            features_df = pd.concat([features_df, current_features_df])

        # Move to the next time interval
        current_time += time_interval  # armazenar o novo tempo para iterar
        features_df.index.name = "time_frame_end"
    # Output the resulting DataFrame with the features for each second
    features_df.index = index
    features_df.index.name = "time_frame_end"

    return features_df


## 0.Funções Para plotagem gráfica


### 0.1 Função para plotar visualizações dataset original
def plot_df(dataframe, x_axis):  # ex: plot_df (df,'adj_time')
    data = dataframe

    fig, ax = plt.subplots(2, 2, figsize=(30, 10))

    selected_columns = data.columns[0:4]  # ajustar aqui quais colunas quer plotar
    for i, column in enumerate(selected_columns):
        sns.lineplot(data=data, x=data[x_axis], y=column, ax=ax.ravel()[i])

    return plt.show()


### 0.2 Função para plotar visualizações dataset de features
def plot_df_features(
    dataframe, x_axis, variavel
):  # ex: plot_df (df,df.index,'Input Torque')
    data = dataframe
    intervalo = {
        "Input Torque": slice(0, 10),
        "gearbox_vibration_x": slice(10, 20),
        "gearbox_vibration_y": slice(20, 30),
        "gearbox_vibration_z": slice(30, 40),
    }

    # para caso de querer plotar o index como x_axis  --> chamar a função plot_df(df,df.index)
    if x_axis is dataframe.index:
        data["index"] = dataframe.index
        x_axis = "index"

    fig, ax = plt.subplots(5, 2, figsize=(30, 20))

    selected_columns = data.columns[
        intervalo[variavel]
    ]  # ajustar aqui quais colunas quer plotar
    for i, column in enumerate(selected_columns):
        sns.lineplot(data=data, x=data[x_axis], y=column, ax=ax.ravel()[i])


### 0.3 Função para comparar datasets
def compare_df_plots(df1, df2, variavel):
    intervalo = {
        "Input Torque": slice(0, 10),
        "gearbox_vibration_x": slice(10, 20),
        "gearbox_vibration_y": slice(20, 30),
        "gearbox_vibration_z": slice(30, 40),
    }

    fig, ax = plt.subplots(
        5, 2, figsize=(30, 20)
    )  # Create the subplot structure outside
    selected_columns = df1.columns[intervalo[variavel]]

    for i, column in enumerate(selected_columns):
        # For df1
        sns.lineplot(
            data=df1,
            x=df1.index,
            y=column,
            ax=ax.ravel()[i],
            color="blue",
            label=f"{'df1'}:{column}",
        )
        # For df2
        sns.lineplot(
            data=df2,
            x=df2.index,
            y=column,
            ax=ax.ravel()[i],
            color="red",
            label=f"{'df2'}:{column}",
        )
        ax.ravel()[i].legend()

    plt.show()


## 4 Funções Basona features


### 4.1 Subfunção para criar basona
def extract_datasets_and_process_features(
    data_dir, fault, intervalo, timeframe
):  # aqui defino no input qual o diretório de data (pasta) e qual função vou usar
    combined_dataframes = []
    combined_datafiles = []
    # Para cada arquivo CSV no diretório
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)

            # Carregando o dataset
            df = pd.read_csv(filepath)

            # Extraindo informações do nome do arquivo e adicionando como novas colunas
            info = extract_info_from_filename(
                filename, fault
            )  ## utilizar a função adequada a cada nome de base ##
            for col, value in info.items():
                df[col] = value
            # Operações tratamento features
            # Chamada Função 1.5
            df = format_dataframe_basona(df)
            # Até aqui o number of files é um valor válido

            # Chamada Função 2.0
            df = format_time(df, intervalo)  # filtragem do tempo
            # Chamada Função 3.0
            # df = features_extract(df)      #### Usarei essa para criar as regras do meu Sistema Especialista (SE) ####
            # Aqui temos a tabela com as features médias do intervalo (Meu SE)

            # Chamada função 3.2
            df = features_timeframe(
                df, timeframe
            )  ###### Escolher tamanho timeframe (número de linhas)########

            # adicionando novamente o nome do arquivo
            for col, value in info.items():
                df[col] = value

            # Adicionando ao conjunto de dataframes combinados
            # col = "Fault"       # fui obrigado a atribuir um valor pra col, pq ele n tava lendo na hr de passar a função miss_teeth_torque, por algum motivo que n sei; teoricamente isso n muda nada no código, só deixa ele um pouco menos genérico, mas como todos meus datasets vão ter a coluna 'Fault' não será problema
            if "Fault" in df.columns:
                if df["Fault"].notna().any():
                    combined_dataframes.append(df)
                    combined_datafiles.append(filename)
    else:
        print(
            f"Columns [col] not found in [filename]"
        )  # essa parte da formula me permite ler pastas com ambos arquivos speed e torque e só pegar o válido e ignorar os demais

    # Combinando todos os dataframes em um único DataFrame final
    # final_df = pd.concat(combined_dataframes, ignore_index=True)
    if combined_dataframes:
        final_df = pd.concat(combined_dataframes, ignore_index=True)
    else:
        final_df = pd.DataFrame()  # Empty DataFrame if no data available

    return final_df, combined_datafiles


# Processando e combinando os datasets
# combined_dataset, file_names = process_and_combine_datasets(teste_data_dir,extract_info_from_filename_speed)


### 4.2 Função para ler 1 dataset
def read_1file_datasets(diretorio, fault):
    dir = f"{diretorio+fault}/"
    combined_dataset, file_names = extract_datasets_and_process_features(
        dir, fault, "inicial", 1
    )
    print(f"number of files = {len(file_names)}")
    print(f"combined files : {file_names}")
    return combined_dataset


### 4.3 Função para concatenar datasets features todos
# Função geral de combinação dos arquivos features
def combine_features_datasets(
    file, intervalo, timeframe
):  # ('treino','inicial','0.1') file = "treino","todos","teste"
    diretorio = "C:/Users/tiago/OneDrive/Desktop/TCC/Dados/split-dataset-2000rpm/"

    dir_geral = f"{diretorio}{file}/geral/"
    combined_dataset_geral, file_names_geral = extract_datasets_and_process_features(
        dir_geral, "geral", intervalo, timeframe
    )
    print(f"number of files geral = {len(file_names_geral)}")
    print(f"combined files geral: {file_names_geral}")

    dir_health = f"{diretorio}{file}/health/"
    combined_dataset_health, file_names_health = extract_datasets_and_process_features(
        dir_health, "health", intervalo, timeframe
    )
    print(f"number of files health = {len(file_names_health)}")
    print(f"combined files health: {file_names_health}")

    dir_misst = f"{diretorio}{file}/miss_teeth/"
    combined_dataset_misst, file_names_misst = extract_datasets_and_process_features(
        dir_misst, "miss_teeth", intervalo, timeframe
    )
    print(f"number of files miss_teeth = {len(file_names_misst)}")
    print(f"combined files miss_teeth: {file_names_misst}")

    comb_datasets = [
        combined_dataset_geral,
        combined_dataset_health,
        combined_dataset_misst,
    ]
    df_concat = pd.concat(comb_datasets, ignore_index=True)
    return df_concat


## 0.4 Plotagem comparação gráfica features health vs falha
def plot_health_non_health_features(
    df_features, torque
):  # torque = "10Nm" #df_features = combine_features_datasets("todos","inicial",1)
    df_features_torque = df_features[df_features["Torque"] == torque]

    # Filter the data based on the 'Fault' column
    health_data = df_features_torque[df_features_torque["Fault"] == "health"]
    non_health_data = df_features_torque[df_features_torque["Fault"] != "health"]

    # List of feature columns to plot
    features = df_features_torque.columns.drop(
        ["Fault", "Degree", "Categoria", "Torque", "Rotação"]
    )  # Assuming 'Fault' is the only non-numeric column

    # Create histograms for each feature
    fig, axs = plt.subplots(len(features), figsize=(16, 5 * len(features)))

    for i, feature in enumerate(features):
        bins = np.linspace(
            min(df_features_torque[feature].min(), non_health_data[feature].min()),
            max(df_features_torque[feature].max(), non_health_data[feature].max()),
            100,
        )
        axs[i].hist(health_data[feature], bins, alpha=0.5, label="health")
        axs[i].hist(non_health_data[feature], bins, alpha=0.5, label="non-health")
        axs[i].legend(loc="upper right")
        axs[i].set_title(feature)

    plt.tight_layout()
    plt.show()
