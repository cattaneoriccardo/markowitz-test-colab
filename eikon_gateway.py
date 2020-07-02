import eikon as ek
import dateutil.relativedelta as rd
import datetime as dt
import numpy as np
import time


def retrieve_monthly_assets_total_returns(date_from, date_to, assets_ric_list):
    if date_from >= date_to:
        raise Exception("Dates are wrong")
    if len(assets_ric_list) == 0:
        raise Exception("At least one RIC needed in the RICs list")

    ek.set_app_key("94baceedc0c24ae8961456ae2b67bc72532e997f")
    start_date = date_from
    end_date = date_to
    timeline = []

    next_from_date = start_date.replace(day=1)
    next_to_date = (start_date + rd.relativedelta(months=1)) + rd.relativedelta(days=-1)
    next_time_slot = (next_from_date, next_to_date)
    while next_time_slot[1] < end_date:
        timeline.append(next_time_slot)
        next_from_date = timeline[-1][1] + rd.relativedelta(days=1)
        next_to_date = (next_from_date + rd.relativedelta(months=1)) + rd.relativedelta(days=-1)
        next_time_slot = (next_from_date, next_to_date)

    total_return_percentage_all = {}
    for target_asset in assets_ric_list:
        total_return_percentage_all[target_asset] = {}
        total_return_percentage_all[target_asset]["timeline"] = []

    for timeslot in timeline:
        start_time = time.time()
        i = timeline.index(timeslot)
        tl = len(timeline)
        f = dt.datetime.strftime(timeslot[0], "%Y-%m-%d")
        t = dt.datetime.strftime(timeslot[1], "%Y-%m-%d")
        o = ek.get_data(assets_ric_list, ['TR.TotalReturn'], {'SDate': f, 'EDate': t})
        o_assets_list = o[0]["Instrument"].to_list()
        for target_asset in assets_ric_list:
            index_of_target_asset = o_assets_list.index(target_asset)
            try:
                to_store = ((f, t), float(o[0]["Total Return"][index_of_target_asset]) / 100)
            except ValueError:
                to_store = ((f, t), np.nan)
            total_return_percentage_all[target_asset]["timeline"].append(to_store)
        end_time = time.time()
        msg = "Loading from Eikon ({:d} of {:d}, {:.2f}%) (remaining time: {:.2f} seconds)"
        print(msg.format(i, tl, 100 * i / tl, (end_time - start_time)*(tl-i)))

    return total_return_percentage_all
