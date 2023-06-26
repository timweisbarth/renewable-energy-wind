import pandas as pd
import xarray


def load_kelmarsh_data(from_raw):
    """ Load the data of the second turbine of kelmarsh wind farm from between
    2016 and 2021.(=all available data of this turbine)

    Returns
    -------
    pandas.DataFrame
        Containing the data
    """
    if from_raw:
        kwf2_paths = [
            "../data/raw/uk/Kelmarsh_SCADA_2016_3082/Turbine_Data"
            + "_Kelmarsh_2_2016-01-03_-_2017-01-01_229.csv",
            "../data/raw/uk/Kelmarsh_SCADA_2017_3083/Turbine_Data"
            + "_Kelmarsh_2_2017-01-01_-_2018-01-01_229.csv",
            "../data/raw/uk/Kelmarsh_SCADA_2018_3084/Turbine_Data_"
            + "Kelmarsh_2_2018-01-01_-_2019-01-01_229.csv",
            "../data/raw/uk/Kelmarsh_SCADA_2019_3085/Turbine_Data_"
            + "Kelmarsh_2_2019-01-01_-_2020-01-01_229.csv",
            "../data/raw/uk/Kelmarsh_SCADA_2020_3086/Turbine_Data_"
            + "Kelmarsh_2_2020-01-01_-_2021-01-01_229.csv",
            "../data/raw/uk/Kelmarsh_SCADA_2021_3087/Turbine_Data_"
            + "Kelmarsh_2_2021-01-01_-_2021-07-01_229.csv"
        ]

        kwf2 = pd.DataFrame()

        for path in kwf2_paths:

            df = pd.read_csv(
                path,
                skiprows=9
            )
            kwf2 = pd.concat([kwf2, df])

        kwf2.to_csv("../data/processed/kelmarsh2_preprocessed.csv")
    else:
        kwf2 = pd.read_csv("../data/processed/kelmarsh2_preprocessed.csv")

    return kwf2


def flatten_multiindex(df, features_with_height_dim, features_with_range_dim):
    """ Flatten the multiindex rows of the df by generating new columns. 

        Example:
        --------
        The function works in such a way that e.g.

            |time_index  | height_index  | feature_vals ...
            |----------------------------------
            |1           |1              |100
            |            |2              |200
            |2           |1              |120
            |            |2              |220
            ...

        results in 

            |time_index  | feature_vals1  | feature_vals2
            |----------------------------------
            |1           |100            |200
            |2           |120            |220
            ...

        Since range & height dimensions are not binary as in this example,
        they will be split in two equally sized buckets before flattening


    Parameters
    ----------
    df: pandas.DataFrame
        Contains the multiindex df
    features_with_height_dim: list
        features of the df that contain a time and height dim
    features_with_range_dim: list
        features of the df that contain a time and range dim


    Returns
    -------
    pandas.DataFrame
        Containing the flattened df with a single datetime index
    """

    # Does the given df have features with an additional height dim?
    if features_with_height_dim:
        # Two buckets of height measurments
        df_height1 = df.where(df["Height"] < 50)[features_with_height_dim]
        df_height2 = df.where(df["Height"] >= 50)[features_with_height_dim]
        df_height1 = df_height1.groupby(level=0).mean()
        df_height2 = df_height2.groupby(level=0).mean()

        # Rename columns
        df_height1.columns = [column_name +
                              "1" for column_name in features_with_height_dim]
        df_height2.columns = [column_name +
                              "2" for column_name in features_with_height_dim]

        df_height = pd.merge(df_height1, df_height2,
                             left_index=True, right_index=True)

    # Does the given df have features with an additional range dim?
    if features_with_range_dim:
        # Two buckets of range measurements
        df_range1 = df.where(df["Range"] < 12)[features_with_range_dim]
        df_range2 = df.where(df["Range"] >= 12)[features_with_range_dim]
        df_range1 = df_range1.groupby(level=0).mean()
        df_range2 = df_range2.groupby(level=0).mean()

        # Rename columns
        df_range1.columns = [column_name +
                             "1" for column_name in features_with_range_dim]
        df_range2.columns = [column_name +
                             "2" for column_name in features_with_range_dim]

        df_range = pd.merge(df_range1, df_range2,
                            left_index=True, right_index=True)

    # Remaining part of df
    df_rest = df.groupby(level=0).mean()
    features_without_extra_dim = list(set(list(df.columns))
                                      - (set(features_with_height_dim).union(set(features_with_range_dim))))
    df_rest = df_rest[features_without_extra_dim]

    # TODO: Caution, test whether Height dim gets eliminated in all cases
    if features_with_height_dim:
        df_combined = pd.merge(
            df_rest, df_height, left_index=True, right_index=True).drop(columns=["Height"])
    if features_with_range_dim:
        df_combined = pd.merge(df_combined, df_range, left_index=True, right_index=True).drop(
            columns=["Range", "range", "Turbine"])

    return df_combined


def load_uebb_data(from_raw):
    """ Load the data of the first turbine of uebb wind farm from between
    2013 and 2014 (=all available data of this turbine).

    Parameters
    ----------
    from_raw : bool
        If false, data will be loaded from csv created last time 
        load_uebb_data(True) was run

    Returns
    -------
    pandas.DataFrame
        Containing the data
    """

    if from_raw:
        # Load .nc file and convert to dataframe
        # TODO: Adjust to ueps, set index necessary?
        uebb = xarray.open_dataset('../data/raw/brazil/UEBB_v1.nc')
        uebb = uebb.to_dataframe()

        # Adjust index
        uebb = uebb.reset_index()
        uebb = uebb.set_index("Time")
        uebb.index = pd.to_datetime(uebb.index)

        # Select turbine
        uebb = uebb.loc[uebb["Turbine"] == 1.0].drop(columns=["Turbine"])

        # Which features have an additional dimension?
        features_with_height_dim = ["wind_speed", "wind_direction", "wind_speed_std", "wind_direction_std", "wind_speed_max",
                                    "wind_speed_min", "wind_speed_cube", "air_temperature", "relative_humidity"]

        uebb = flatten_multiindex(uebb, features_with_height_dim, [])

        uebb.to_csv("../data/processed/uebb_preprocessed.csv")
    else:
        uebb = pd.read_csv("../data/processed/uebb_preprocessed.csv")

    return uebb


def load_ueps_data(from_raw):
    """ Load the data of the second turbine of ueps wind farm from between
    2013 and 2014 (=all available data of this turbine).

    Parameters
    ----------
    from_raw : bool
        If false, data will be loaded from csv created last time 
        load_ueps_data(True) was run

    Returns
    -------
    pandas.DataFrame
        Containing the data
    """
    if from_raw:

        # Load .nc file and convert to dataframe
        ueps = xarray.open_dataset('../data/raw/brazil/UEPS_v1.nc')
        ueps = ueps.isel(Turbine=2)
        ueps = ueps.to_dataframe()
        ueps = ueps.reset_index(["Height", "Range"])

        # Which features have an additional dimension?
        features_with_range_dim = ["lidar_wind_speed", "lidar_wind_direction", "lidar_wind_speed_std", "lidar_ws_u",
                                   "lidar_ws_v", "lidar_ws_w", "lidar_availability"]
        features_with_height_dim = ["wind_speed", "wind_direction", "wind_speed_std", "wind_direction_std", "wind_speed_max",
                                    "wind_speed_min", "wind_speed_cube", "air_temperature", "relative_humidity",
                                    "UST", "UST_flag", "HS", "HS_flag", "TKE", "LMO"]

        ueps = flatten_multiindex(
            ueps, features_with_height_dim, features_with_range_dim)

        ueps.to_csv("../data/processed/ueps_preprocessed.csv")
    else:
        ueps = pd.read_csv("../data/processed/ueps_preprocessed.csv")

    return ueps
