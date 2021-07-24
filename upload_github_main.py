from operator import index
import random
import re
from datetime import datetime, timedelta
from pprint import pprint
from time import time as now
from typing import List

import os
import numpy as np
import pandas as pd

if os.name == 'nt':
    USE_PANDARALLEL = False
else:
    try:
        from pandarallel import pandarallel
        pandarallel.initialize()
        USE_PANDARALLEL = True
    except ImportError:
        USE_PANDARALLEL = False

SEED = 0
DEFAULT_SEGMEN = 6
DATASET_PATH = './scenario_dataset_1/dataset_result.binetflow'


OUTPUT_EXTENSION = 'binetflow'
OUTPUT_NAME = 'dataset_result'


np.random.seed(SEED)
random.seed(SEED)


def __rand_sec(count: int, scale: int = 3599):
    desired_mean = scale/2
    desired_std_dev = scale/4

    samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=count)
    # pprint(samples)

    samples = samples - (np.mean(samples))

    samples = samples * (desired_std_dev/np.std(samples))

    samples = samples + desired_mean

    samples = np.rint(samples)

    samples = np.abs(samples)

    samples = samples - 3600

    samples = np.abs(samples)

    return list(map(int, list(samples)))


def __gen_datetime(start: datetime, delta: List[int]):
    return [start + timedelta(seconds=s) for s in delta]


def distribute(sum: int, count: int):
    desired_mean = sum/count
    desired_std_dev = desired_mean/10

    samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=count)

    samples = samples - (np.mean(samples))

    samples = samples * (desired_std_dev/np.std(samples))

    samples = samples + desired_mean

    samples = np.rint(samples)

    diff = np.sum(samples) - sum
    while diff != 0:
        pos = np.random.randint(samples.shape[0])
        samples[pos] = samples[pos] + 1 if diff < 0 else samples[pos] - 1
        diff = np.sum(samples) - sum

    return list(map(int, list(samples)))


def clean_numbers(s: str):
    s = re.sub(r'-\d+', '', s)
    s = re.sub(r'\d+$', '', s)
    s = re.sub(r'-CC\d+', '', s)
    # s = re.sub(r'V\d+', '', s)
    return s


def __input_int(prompt: str = '', default=None):
    try:
        return int(input(prompt))
    except ValueError:
        return default


def __input_str(prompt: str = '', default=None):
    return input(prompt) or default


def main():
    df_path = __input_str(
        f'Path ({DATASET_PATH}) : ',
        DATASET_PATH
    )
    print(f'Load dataset: {df_path}')
    t0 = now()
    df = pd.read_csv(df_path)

    if USE_PANDARALLEL:
        df['Label'] = df['Label'].parallel_apply(clean_numbers)
    else:
        df['Label'] = df['Label'].apply(clean_numbers)

    df_bot = df[df['Label'].str.contains('Botnet_relation')]
    df_normal = df[~df['Label'].str.contains('Botnet_relation')]
    print(f'Load selesai ({now()-t0:0.2f}s)')
    del df

    max_segmen = __input_int(
        f'Waktu (jam) ({DEFAULT_SEGMEN}):',
        DEFAULT_SEGMEN
    )

    bot_count = len(df_bot["SrcAddr"].unique())
    bot_count = __input_int(
        f'Jumlah BOT ({bot_count}): ',
        bot_count
    )

    # pprint(df_bot["Label"].unique())
    flow_type = df_bot.groupby(["SrcAddr"]).nunique()["Label"].min()
    flow_type = __input_int(
        f'Jenis flow ({flow_type}): ',
        flow_type
    )

    flow_count = max_segmen*5
    flow_count = __input_int(
        f'Jumlah per flow ({flow_count}): ',
        flow_count
    )

    normal_count = len(df_normal.index)
    normal_count = __input_int(
        f'Jumlah NORMAL ({normal_count}): ',
        normal_count
    )

    _bot_record_count = bot_count*flow_type*flow_count
    _total_record_count = normal_count + _bot_record_count
    print(f'Normal record: {normal_count} '
          f'({(normal_count/_total_record_count)*100:0.2f}%)')
    print(f'Bot record   : {_bot_record_count} '
          f'({(_bot_record_count/_total_record_count)*100:0.2f}%)')

    input('Tekan ENTER untuk mulai...')

    print('###### START PROCESS ######')

    t0 = now()
    # time for each segmen
    time_bucket = [
        datetime(2020, 1, 1),
    ]
    for _ in range(max_segmen):
        time_bucket.append(time_bucket[-1] + timedelta(hours=1))

    # normal flow distribution in each segmen
    normal_distribution = distribute(normal_count, max_segmen)

    df_segmen = [pd.DataFrame(columns=df_normal.columns)
                 for _ in range(max_segmen)]
    bots = random.sample(list(df_bot['SrcAddr'].unique()), bot_count)
    # pprint(bots)
    bots_flow = {
        b: random.sample(
            list(df_bot[df_bot['SrcAddr'] == b]['Label'].unique()),
            flow_type)
        for b in bots
    }
    # pprint(bots_flow)
    # bot flow distribution in each segmen
    bots_distribution = {
        bot_addr: {
            bot_flow: distribute(flow_count, max_segmen) for bot_flow in bots_flow[bot_addr]
        } for bot_addr in bots_flow
    }

    for s in range(max_segmen):
        for _addr, _flows in bots_distribution.items():
            for _flow, _dist in _flows.items():
                _df = df_bot[(df_bot['SrcAddr'] == _addr)
                             & (df_bot['Label'] == _flow)]
                # print(len(_df.index))
                _df = _df.sample(_dist[s], replace=True,
                                 random_state=random.randint(0, 100000))
                df_segmen[s] = pd.concat([df_segmen[s], _df])
                del _df

        df_segmen[s] = pd.concat([
            df_segmen[s],
            df_normal.sample(normal_distribution[s], replace=True,
                             random_state=random.randint(0, 100000))
        ])

        _sec_list = __rand_sec(len(df_segmen[s].index))
        df_segmen[s] = df_segmen[s].sample(frac=1.0,
                                           random_state=random.randint(0, 100000))
        df_segmen[s]['StartTime'] = __gen_datetime(time_bucket[s], _sec_list)

    df_result = pd.concat(df_segmen)
    df_result.sort_values(by='StartTime', inplace=True)
    df_result.reset_index(drop=True, inplace=True)
    # print(df_result)
    df_result.to_csv(f'{OUTPUT_NAME}.{OUTPUT_EXTENSION}', index=False)
    df_result.to_csv(f'{OUTPUT_NAME}.without_label.{OUTPUT_EXTENSION}',
                     index=False,
                     columns=df_result.columns[:-1])

    df_result[df_result['Label'].str.contains('Botnet')].to_csv(
        f'{OUTPUT_NAME}.botnet_only.{OUTPUT_EXTENSION}',
        index=False
    )

    df_result[~df_result['Label'].str.contains('Botnet_chain')].to_csv(
        f'{OUTPUT_NAME}.normal_only.{OUTPUT_EXTENSION}',
        index=False
    )
    print(f'Runtime      : {now()-t0:0.2f}s')
    print(f'Output       : {OUTPUT_NAME}')
    print(f'Description  : ')
    print(f'Start        : {time_bucket[0]}')
    print(f'Finish       : {time_bucket[-1]}')
    print(f'Durasi       : {max_segmen} jam')
    print(f'Total record : {len(df_result.index)}')
    print(f'Feature      : {len(df_result.columns)}')
    print(f'  ', end='')
    for f in list(df_result.columns):
        print(f'{f} ', end='')
    print('')
    print(f'Host         : {len(df_result["SrcAddr"].unique())}')
    print(f'Bots         : {len(df_bot["SrcAddr"].unique())}')
    for ip in list(df_bot["SrcAddr"].unique()):
        print(f'    {ip} :: '
              f'{len(df_result[df_result["SrcAddr"] == ip].index)} :: '
              f'{df_result[df_result["SrcAddr"] == ip]["StartTime"].iloc[0]} - '
              f'{df_result[df_result["SrcAddr"] == ip]["StartTime"].iloc[-1]}')

    print(f'Bot record   : {_bot_record_count} '
          f'({(_bot_record_count/_total_record_count)*100:0.2f}%)')
    print(f'Normal record: {normal_count} '
          f'({(normal_count/_total_record_count)*100:0.2f}%)'
          f'\n    Host     : {len(df_result[~df_result["Label"].str.contains("Botnet_chain")]["SrcAddr"].unique())}')

    # Start time + End Time

# OUTPUT to txt
    with open(f'{OUTPUT_NAME}.description.txt', 'w') as outfile:
        print(f'Start        : {time_bucket[0]}', file=outfile)
        print(f'Finish       : {time_bucket[-1]}', file=outfile)
        print(f'Durasi       : {max_segmen} jam', file=outfile)
        print(f'Total record : {len(df_result.index)}', file=outfile)
        print(f'Feature      : {len(df_result.columns)}', file=outfile)
        print(f'  ', end='', file=outfile)
        for f in list(df_result.columns):
            print(f'{f} ', end='', file=outfile)
        print('', file=outfile)
        print(
            f'Host         : {len(df_result["SrcAddr"].unique())}', file=outfile)
        print(
            f'Bots         : {len(df_bot["SrcAddr"].unique())}', file=outfile)
        for ip in list(df_bot["SrcAddr"].unique()):
            print(f'    {ip} :: '
                  f'{len(df_result[df_result["SrcAddr"] == ip].index)} :: '
                  f'{df_result[df_result["SrcAddr"] == ip]["StartTime"].iloc[0]} - '
                  f'{df_result[df_result["SrcAddr"] == ip]["StartTime"].iloc[-1]}', file=outfile)

        print(f'Bot record   : {_bot_record_count} '
              f'({(_bot_record_count/_total_record_count)*100:0.2f}%)', file=outfile)
        print(f'Normal record: {normal_count} '
              f'({(normal_count/_total_record_count)*100:0.2f}%)'
              f'\n    Host     : {len(df_result[~df_result["Label"].str.contains("Botnet_chain")]["SrcAddr"].unique())}', file=outfile)


if __name__ == "__main__":
    main()
