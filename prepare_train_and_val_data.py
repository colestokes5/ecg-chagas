import os
import random
import shutil
import argparse
import sys
from helper_code import load_label, load_text
from sklearn.model_selection import train_test_split

def main(retain_all=False):
    # Paths
    src_dir = 'data/all_data'
    train_dir = 'data/train_data'
    val_dir = 'data/val_data'

    # Make output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Gather all .dat files
    all_files = [f for f in os.listdir(src_dir) if f.endswith('.hea')]
    base_names = [os.path.splitext(f)[0] for f in all_files]

    # Load labels
    labeled_files = []
    for name in base_names:
        label = load_label(os.path.join(src_dir, name))
        labeled_files.append((name, label))

    # Split into positive and negative labels
    pos_files = [name for name, label in labeled_files if label == 1]
    neg_files = [name for name, label in labeled_files if label == 0]

    # Shuffle for randomness
    random.seed(821)
    random.shuffle(neg_files)

    # Balance negative samples
    if retain_all:
        balanced_neg_files = neg_files
    else:
        num_pos = len(pos_files)
        half = num_pos // 2
        ptbxl_neg = [f for f in neg_files if 'PTB-XL' in load_text('data/all_data/' + f + '.hea')][:half]
        code15_neg = [f for f in neg_files if 'CODE-15%' in load_text('data/all_data/' + f + '.hea')][:num_pos - len(ptbxl_neg)]
        balanced_neg_files = ptbxl_neg + code15_neg

    # Final dataset
    final_files = pos_files + balanced_neg_files
    random.shuffle(final_files)

    # Split into train and val
    train_files, val_files = train_test_split(final_files, test_size=0.2, random_state=821)

    def copy_pairs(files, target_dir):
        for name in files:
            for ext in ['.dat', '.hea']:
                src = os.path.join(src_dir, name + ext)
                dst = os.path.join(target_dir, name + ext)
                shutil.copyfile(src, dst)

    copy_pairs(train_files, train_dir)
    copy_pairs(val_files, val_dir)

    print(f"Finished. Train: {len(train_files)}, Val: {len(val_files)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split and balance dataset')
    parser.add_argument('--retain_all', action='store_true', help='Retain all negative files instead of balancing')
    args = parser.parse_args()
    main(retain_all=args.retain_all if '--retain_all' in sys.argv else False)
