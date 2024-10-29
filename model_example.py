import pandas as pd

from torch.utils.data import DataLoader

from typing import Tuple

from exeqt import task, model
from exeqt_udm_interface import UDMBlobClient

from models.model_example.parameters import SimulationParameter, SettingsParameter
from models.utils import udm_io

from models.model_example.fore_utils_gryt import *


@task
def load_input_data(udm_client: UDMBlobClient, settings: SettingsParameter) -> pd.DataFrame:
    """
    Loads the input time-series data from the UDM client based on the given settings.

    Args:
        udm_client (UDMBlobClient): The UDM client used for data retrieval.
        settings (SettingsParameter): Contains information for loading the input data, such as identifier and time zone.

    Returns:
        pd.DataFrame: Loaded time-series data, indexed by 'Time'.
    """

    id = settings.UDM_lake_identifier

    timeseries = udm_client.load_latest_timestamp(identifier=id)

    df = pd.read_parquet(BytesIO(timeseries["data"]))
    print("Data loaded")
    try:
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.set_index("Time")
        print("Data indexed")
    except:
        print("Data indexing failed")

    return df


@task
def load_logs(udm_client: UDMBlobClient, settings: SettingsParameter) -> pd.DataFrame:
    """
    Loads the logs from the UDM client based on the given settings.

    Args:
        udm_client (UDMBlobClient): The UDM client used for log retrieval.
        settings (SettingsParameter): Contains information for loading the logs, such as identifier and time zone.

    Returns:
        pd.DataFrame: Loaded logs.
    """

    id = settings.UDM_log_identifier

    logs = udm_client.load_latest_timestamp(identifier=id)

    df = pd.read_parquet(BytesIO(logs["data"]))
    print("Logs loaded")

    return df


@task
def identify_best_loss(logs: pd.DataFrame) -> float:
    return logs["best_loss"].min()


@task
def create_loader(simulation_parameters: SimulationParameter, timeseries_data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
    """
    Creates a PyTorch DataLoader for the wind speed dataset using the given simulation parameters.

    Args:
        simulation_parameters (SimulationParameter): Settings used for creating the dataset and DataLoader,
        such as window size, steps, and batch size.
        timeseries_data (pd.DataFrame): The time-series data containing the wind speed measurements.

    Returns:
        DataLoader: PyTorch DataLoader with batched and shuffled dataset.
    """
    try:
        train_set = WindSpeedDataset(
            timeseries_data,
            simulation_parameters.window_size,
            simulation_parameters.steps,
            simulation_parameters.stream,
            "train",
        )
        val_set = WindSpeedDataset(
            timeseries_data,
            simulation_parameters.window_size,
            simulation_parameters.steps,
            simulation_parameters.stream,
            "val",
        )

        train_loader = DataLoader(train_set, batch_size=simulation_parameters.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=simulation_parameters.batch_size, shuffle=False)

    except:
        print("Dataset creation failed")

    # Return results
    return train_loader, val_loader


@task
def create_model(
    udm_client: UDMBlobClient, settings: SettingsParameter, simulation_parameters: SimulationParameter
) -> nn.Module:
    """
    Initializes the machine learning model for wind speed forecasting, using parameters for architecture configuration.

    Args:
        simulation_parameters (SimulationParameter): Contains model configurations such as window size, hidden size, and steps.

    Returns:
        nn.Module: Initialized machine learning model, potentially with preloaded state.
    """
    try:
        model = MLP(simulation_parameters.window_size, simulation_parameters.hidden_size, simulation_parameters.steps)

        print("Model created")

        if simulation_parameters.load_model:
            id = settings.UDM_model_identifier

            model_data = udm_client.load_latest_timestamp(identifier=id)

            state_dict = torch.load(BytesIO(model_data["data"]))
            # state_dict = torch.load(simulation_parameters.load_file)
            model.load_state_dict(state_dict)
            print("Model loaded")
        else:
            print("No model to load")

    except:
        print("Model creation failed")

    return model


@task
def train_model(simulation_parameters: SimulationParameter, model: nn.Module, train_loader: DataLoader) -> nn.Module:
    """
    Trains the machine learning model using a specified loss function and optimizer.

    Args:
        simulation_parameters (SimulationParameter): Training configurations like number of epochs and learning rate.
        model (nn.Module): The machine learning model to be trained.
        train_loader (DataLoader): DataLoader providing batches of data for training the model.

    Returns:
        nn.Module: Trained model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(simulation_parameters.n_epochs):
        for i, batch in enumerate(train_loader):
            x, y = batch

            optimizer.zero_grad()

            y_pred = model(x.float())

            loss = criterion(y_pred, y.float())

            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


@task
def validate_model(model: nn.Module, val_loader: DataLoader) -> float:
    """
    Validates the model performance on the validation dataset by computing the average loss.

    Args:
        model (nn.Module): The trained machine learning model.
        val_loader (DataLoader): DataLoader providing validation data.

    Returns:
        float: The average validation loss.
    """
    criterion = torch.nn.MSELoss()

    model.eval()
    val_loss = 0.0
    total_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch

            y_hat = model(x.float())
            loss = criterion(y, y_hat)

            val_loss += loss.item()

    avg_val_loss = val_loss / total_batches

    return avg_val_loss


@task
def get_data(simulation_parameters: SimulationParameter, split: str = "train") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves the data for a specific split (train/test) from the API URL specified in simulation parameters.

    Args:
        simulation_parameters (SimulationParameter): Configuration for fetching data, including API URL and seed.
        split (str): The data split to fetch ('train' or 'test').

    Returns:
        pd.DataFrame: DataFrame containing the retrieved data for the specified split.
    """

    url = simulation_parameters.api_url
    seed = simulation_parameters.set_seed
    window_size = simulation_parameters.window_size
    steps_ahead = simulation_parameters.steps

    if split == "test":
        url = f"{url}/test/{seed}/{window_size}/{steps_ahead}"
    else:
        url = f"{url}/{split}"

    response = requests.get(url)

    if response.status_code == 200:
        if split == "test":
            data = response.json()

            input_data = pd.DataFrame(data["input_data"])
            target_data = pd.DataFrame(data["target_data"])

            input_data["Time"] = pd.to_datetime(input_data["Time"])
            input_data.set_index("Time", inplace=True)

            target_data["Time"] = pd.to_datetime(target_data["Time"])
            target_data.set_index("Time", inplace=True)

            print(f"Test data retrieved for {target_data.iloc[:1].index} split")

            return input_data, target_data
        else:
            data = response.json()
            df = pd.DataFrame(data)
            df["Time"] = pd.to_datetime(df["Time"])
            df.set_index("Time", inplace=True)

            print(f"{split} data retrieved")

            return df
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")


@task
def create_output_data(
    simulation_parameters: SimulationParameter, model: nn.Module, x_test: pd.DataFrame, y_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Uses the trained model to generate predictions for the test dataset and formats the results for output.

    Args:
        simulation_parameters (SimulationParameter): Parameters specifying data stream and forecast window size.
        model (nn.Module): The trained machine learning model used to generate predictions.
        x_test (pd.DataFrame): Input data for the test set.
        y_test (pd.DataFrame): Target data for the test set.

    Returns:
        pd.DataFrame: DataFrame containing the input, target, and predicted values, indexed by time.
    """

    if simulation_parameters.stream is not None:
        x = x_test[simulation_parameters.stream].values
        y = y_test[simulation_parameters.stream].values

    else:
        x = x_test.values
        y = y_test.values

    x_tensor = torch.tensor(x, dtype=torch.float32)

    y_hat = model(x_tensor).detach().numpy()

    combined_time = pd.concat([pd.Series(x_test.index), pd.Series(y_test.index)], ignore_index=True)

    data_dict = {
        "time": combined_time,
        "input": np.concatenate([x.squeeze(), np.full(12, np.nan)]),
        "target": np.concatenate([np.full(24, np.nan), y]),
        "prediction": np.concatenate([np.full(24, np.nan), y_hat.squeeze()]),
    }

    data_df = pd.DataFrame(data_dict)
    data_df["time"] = pd.to_datetime(data_df["time"])
    data_df = data_df.set_index("time")

    return data_df


@task
def append_data(data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Appends new data that is not present in the original dataset.

    Args:
        data (pd.DataFrame): Original dataset.
        new_data (pd.DataFrame): New data to be appended.

    Returns:
        pd.DataFrame: Combined dataset with appended new data.
    """

    new_rows = new_data.loc[~new_data.index.isin(data.index)]

    data = pd.concat([data, new_rows])

    return data


@task
def store_lake(udm_client: UDMBlobClient, results: pd.DataFrame, settings: SettingsParameter) -> None:
    """
    Stores the calculated results or updated dataset into the UDM system.

    Args:
        udm_client (UDMBlobClient): Client used to store results.
        results (pd.DataFrame): DataFrame containing the results to be stored.
        settings (SettingsParameter): Storage configuration parameters such as identifiers and formats.
        lake (bool): Indicates if the updated data lake should be stored.
    """

    try:
        if settings.UDM_lake_identifier != "NO UDM EXPORT":
            udm_io.store_results(
                udm_client,
                results,
                identifier=settings.UDM_lake_identifier,
                time_zone=settings.time_zone,
                double_precision=settings.number_of_decimal_places,
                format=settings.UDM_lake_format,
            )
    except:
        # Raise an exception if the storage fails
        print("Failed to store results")


@task
def store_forecasts(udm_client: UDMBlobClient, forecast: pd.DataFrame, settings: SettingsParameter) -> None:
    """
    Stores the calculated results or updated dataset into the UDM system.

    Args:
        udm_client (UDMBlobClient): Client used to store results.
        results (pd.DataFrame): DataFrame containing the results to be stored.
        settings (SettingsParameter): Storage configuration parameters such as identifiers and formats.
        lake (bool): Indicates if the updated data lake should be stored.
    """

    # try:

    if settings.UDM_forecast_identifier != "NO UDM EXPORT":
        udm_io.store_results(
            udm_client,
            forecast,
            identifier=settings.UDM_forecast_identifier,
            time_zone=settings.time_zone,
            double_precision=settings.number_of_decimal_places,
            format=settings.UDM_forecast_format,
        )
    # except:
    #     # Raise an exception if the storage fails
    #     print("Failed to store results")


@task
def store_model(udm_client: UDMBlobClient, model: pd.DataFrame, settings: SettingsParameter) -> None:
    """
    Stores the trained model into the UDM system.

    Args:
        udm_client (UDMBlobClient): Client used to store the model.
        model (pd.DataFrame): The trained model to be stored.
        settings (SettingsParameter): Model storage settings, such as format and identifiers.
    """
    try:
        if settings.UDM_model_format != "NO UDM EXPORT":
            udm_io.store_results(
                udm_client,
                model,
                identifier=settings.UDM_model_identifier,
                time_zone=settings.time_zone,
                double_precision=settings.number_of_decimal_places,
                format=settings.UDM_model_format,
            )

    except:
        # Raise an exception if the storage fails
        print("Failed to store model")


@task
def store_logs(udm_client: UDMBlobClient, logs: pd.DataFrame, settings: SettingsParameter) -> None:
    """
    Stores the logs into the UDM system.

    Args:
        udm_client (UDMBlobClient): Client used to store the logs.
        logs (pd.DataFrame): The logs to be stored.
        settings (SettingsParameter): Log storage settings, such as format and identifiers.
    """
    # try:
    if settings.UDM_log_format != "NO UDM EXPORT":
        udm_io.store_results(
            udm_client,
            logs,
            identifier=settings.UDM_log_identifier,
            time_zone=settings.time_zone,
            double_precision=settings.number_of_decimal_places,
            format=settings.UDM_log_format,
        )

    # except:
    # Raise an exception if the storage fails
    # print("Failed to store logs")


@model(name="Weather Forecasting", owner="PSRG AI Weather", description="model_example.md")
def model_example(simulation_parameter: SimulationParameter, settings: SettingsParameter):
    if simulation_parameter.init_data:
            
            udm_client = UDMBlobClient()
    
            data = pd.read_parquet('models/model_example/init_resources/sere_training_2018_2021.parquet')

            model = MLP(simulation_parameter.window_size, simulation_parameter.hidden_size, simulation_parameter.steps)
            model.load_state_dict(torch.load('models/model_example/init_resources/init_model_weights.pth'))
    
            store_lake(udm_client, data, settings)

            store_model(udm_client, model, settings)
    
            logs = pd.DataFrame(
                {
                    "time": [pd.Timestamp.now()],
                    "Validation Loss": [100],
                    "best_loss": [100],
                    "lake_size": [len(data)],
                    "retrained": [False],
                    "seed" : [simulation_parameter.set_seed]
                }
            )
    
            store_logs(udm_client, logs, settings)

    else:

        ### Load historical data

        udm_client = UDMBlobClient()

        # Load input data - Training data
        timeseries_data = load_input_data(udm_client, settings)

        logs = load_logs(udm_client, settings)

        best_loss = identify_best_loss(logs)

        ### Create model
        model = create_model(udm_client, settings, simulation_parameter)


        ### Create loaders
        future = create_loader(simulation_parameter, timeseries_data)

        train_loader = future[0]
        val_loader = future[1]

        ### Train model
        if simulation_parameter.init_train:
            model = train_model(simulation_parameter, model, train_loader)

        ### Validate model
        val_loss = validate_model(model, val_loader)

        retrained = False

        if simulation_parameter.allow_retrain and val_loss > (best_loss * 1.5):
            print("Retraining model and storing model.")
            model = train_model(simulation_parameter, model, train_loader)
            store_model(udm_client, model, settings)

            retrained = True

        ### Store model
        if simulation_parameter.store_model and val_loss < best_loss:
            print("Storing model")
            store_model(udm_client, model, settings)
            best_loss = val_loss

        ### Predict using the model
        future = get_data(simulation_parameter, split="test")

        x_test = future[0]
        y_test = future[1]
        
        forecast = create_output_data(simulation_parameter, model, x_test, y_test)

        ### Store results
        store_forecasts(udm_client, forecast, settings)

        ### Append and store new data to lake

        updated_lake = append_data(timeseries_data, x_test)
        store_lake(udm_client, updated_lake, settings)

        ### Add logs

        add_logs = pd.DataFrame(
            {
                "time": [pd.Timestamp.now()],
                "Validation Loss": [val_loss],
                "best_loss": [best_loss],
                "lake_size": [len(updated_lake)],
                "retrained": [retrained],
                "seed" : [simulation_parameter.set_seed]
            }
        )

        logs = pd.concat([logs, add_logs], ignore_index=True)
        store_logs(udm_client, logs, settings)

    return logs
