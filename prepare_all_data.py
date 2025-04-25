import subprocess
import os

def run_prepare_samitrop_data():
    command = [
        'python', 'prepare_samitrop_data.py',
        '-i', 'data/input_data/samitrop_exams.hdf5',
        '-d', 'data/input_data/samitrop_exams.csv',
        '-l', 'data/input_data/samitrop_chagas_labels.csv',
        '-o', 'data/all_data'
    ]
    subprocess.run(command)

def run_prepare_ptbxl_data():
    command = [
        'python', 'prepare_ptbxl_data.py',
        '-i', 'data/input_data/ptbxl_files',
        '-d', 'data/input_data/ptbxl_database.csv',
        '-o', 'data/all_data'
    ]
    subprocess.run(command)

def run_prepare_code15_data():
    for i in range(18):  # Looping from 0 to 17
        command = [
            'python', 'prepare_code15_data.py',
            '-i', f"data/input_data/code15_files/exams_part{i}.hdf5",
            '-d', 'data/input_data/code15_files/exams.csv',
            '-l', 'data/input_data/code15_chagas_labels.csv',
            '-o', 'data/all_data'
        ]
        subprocess.run(command)

if __name__ == "__main__":
    print('Processing Samitrop...')
    run_prepare_samitrop_data()
    print('Processing PTBXL...')
    run_prepare_ptbxl_data()
    print('Processing Code15...')
    run_prepare_code15_data()
    print('Complete')
