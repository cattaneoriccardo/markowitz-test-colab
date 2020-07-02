import datetime as dt
import eikon_gateway as eg
import json


def load_data_from_file(file_name):
    with open(file_name, 'r') as infile:
        return json.load(infile)


def load_data_from_eikon(start_date, end_date, target_assets):
    # consume Eikon
    ek_data = eg.retrieve_monthly_assets_total_returns(start_date, end_date, target_assets)
    ek_data_processed = dict()
    for asset_name in target_assets:
        ek_data_processed[asset_name] = [x[1] for x in ek_data[asset_name]["timeline"]]
    ek_data_processed["start_date"] = dt.date.strftime(start_date, "%Y-%m-%d")
    ek_data_processed["end_date"] = dt.date.strftime(end_date, "%Y-%m-%d")
    ek_data_processed["target_assets"] = target_assets
    return ek_data_processed
