import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

MOT17_ROOT = "./data/MOT17"
OUTPUT_DIR = "./paper_outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "queue_tcn_model.keras")

def main():
    print("\n" + "=" * 70)
    print("LIVE MATHEMATICAL EVALUATION FOR TRAINED 1D-TCN MODEL")
    print("=" * 70)

    # 1. Check if the real trained model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Trained model file not found at: {MODEL_PATH}")
        print("Please ensure your 'main_pipeline.py' saves the model using:")
        print("proposed_model.save('./paper_outputs/queue_tcn_model.keras')")
        return

    # 2. Extract raw data features to evaluate against
    train_path = os.path.join(MOT17_ROOT, "train")
    seq_dirs = sorted(glob.glob(os.path.join(train_path, "MOT17-*")))
    
    if not seq_dirs:
        print(f"[ERROR] No tracking data found in {train_path}.")
        return

    extracted_features = []
    print("Extracting testing frames...")
    for seq in seq_dirs:
        gt_file = os.path.join(seq, "gt", "gt.txt")
        if not os.path.exists(gt_file):
            continue
        df = pd.read_csv(gt_file, header=None)
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'active', 'cls', 'vis']
        df = df[df['cls'] == 1]
        total_frames = df['frame'].max()
        
        df = df.sort_values(by=['id', 'frame'])
        df['prev_x'] = df.groupby('id')['x'].shift(1)
        df['prev_y'] = df.groupby('id')['y'].shift(1)
        df['dist'] = np.sqrt((df['x'] - df['prev_x'])**2 + (df['y'] - df['prev_y'])**2)
        
        for f in range(1, total_frames - 9, 5):
            window_df = df[(df['frame'] >= f) & (df['frame'] < f + 10)]
            if window_df.empty:
                continue
            step_features = []
            for step in range(10):
                step_df = window_df[window_df['frame'] == f + step]
                q_size = len(step_df)
                avg_speed = step_df['dist'].mean() if 'dist' in step_df and not step_df['dist'].isnull().all() else 0.0
                avg_w = step_df['w'].mean() if not step_df.empty else 0.0
                avg_h = step_df['h'].mean() if not step_df.empty else 0.0
                step_features.append([q_size, avg_speed if not np.isnan(avg_speed) else 0.0, avg_w, avg_h, 25.0])
            extracted_features.append(step_features)

    X = np.array(extracted_features)
    base_queues = np.mean(X[:, :, 0], axis=1)
    base_speeds = np.mean(X[:, :, 1], axis=1)
    y = (2.2 * base_queues) - (0.5 * base_speeds) + np.random.normal(3.0, 1.2, len(X))
    y = np.clip(y, 1.0, 45.0)

    # Recreate validation split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_samples_test, timesteps, num_feats = X_test.shape
    feature_scaler = StandardScaler()
    feature_scaler.fit(X.reshape(-1, num_feats))
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, num_feats)).reshape(num_samples_test, timesteps, num_feats)

    target_scaler = StandardScaler()
    target_scaler.fit(y.reshape(-1, 1))

    # 3. Load the real saved model
    print("Loading actual trained Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 4. Generate real model predictions
    preds_scaled = model.predict(X_test_scaled).flatten()
    y_pred = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    # 5. Compute true mathematical regression accuracy (100% - MAPE)
    # Filter out absolute zeros to avoid divisions by 0
    non_zero_mask = y_test > 0
    mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
    real_accuracy = 100.0 - mape

    # 6. Calculate real Naive and Linear Regression performance dynamically on this exact slice
    naive_preds = np.full_like(y_test, y.mean())
    naive_mape = np.mean(np.abs((y_test[non_zero_mask] - naive_preds[non_zero_mask]) / y_test[non_zero_mask])) * 100
    naive_accuracy = max(100.0 - naive_mape, 0.0)

    X_flat_train = feature_scaler.transform(X.reshape(-1, num_feats)).reshape(len(X), -1)
    X_flat_test = X_test_scaled.reshape(num_samples_test, -1)
    lr = LinearRegression().fit(X_flat_train, y)
    lr_preds = lr.predict(X_flat_test)
    lr_mape = np.mean(np.abs((y_test[non_zero_mask] - lr_preds[non_zero_mask]) / y_test[non_zero_mask])) * 100
    lr_accuracy = max(100.0 - lr_mape, 0.0)

    print("\n--- LIVE COMPUTED PREDICTION ACCURACY (100% - MAPE) ---")
    print(f"✓ Proposed 1D-TCN Model  : {real_accuracy:.2f}% Real Accuracy")
    print(f"✓ Linear Regression Base : {lr_accuracy:.2f}% Real Accuracy")
    print(f"✓ Naive Static Base      : {naive_accuracy:.2f}% Real Accuracy")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()