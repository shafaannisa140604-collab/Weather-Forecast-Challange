import pandas as pd
import numpy as np
import os

class MarkovPrecipitationModel:
    def __init__(self, num_bins=10):
        self.num_bins = num_bins
        self.bins = np.linspace(0, 100, num_bins + 1)
        self.transition_matrix = None
        self.state_centers = (self.bins[:-1] + self.bins[1:]) / 2

    def _discretize(self, data):
        """Converts continuous probabilities to discrete states (bins)."""
        # np.digitize returns indices 1 to len(bins)-1. We want 0-indexed states.
        # We subtract 1 to get 0-indexed states.
        # Values exactly 0 need to be handled to stay in bin 0.
        indices = np.digitize(data, self.bins, right=True)
        # digitize with right=True: bins[i-1] < x <= bins[i]
        # But we want 0..10 -> bin 0
        
        # Let's use pd.cut for easier labeling
        labels = range(self.num_bins)
        return pd.cut(data, bins=self.bins, labels=labels, include_lowest=True).astype(int)

    def fit(self, df):
        """
        Fits the Markov chain model on the dataset.
        Assumes df has 'region', 'time', and 'precipitation_probability (%)'.
        """
        print("Preprocessing data...")
        # Ensure sorted chronologically per region
        df = df.sort_values(by=['region', 'time'])
        
        # Discretize precipitation probability
        df['state'] = self._discretize(df['precipitation_probability (%)'])
        
        # Create transition pairs
        transitions = []
        
        print("Building transition matrix...")
        # Iterate by region to ensure we don't transition across disconnected regions
        for _, group in df.groupby('region'):
            states = group['state'].values
            # Zip current state with next state
            transitions.extend(zip(states[:-1], states[1:]))
            
        # Count transitions
        transition_counts = np.zeros((self.num_bins, self.num_bins))
        for current_state, next_state in transitions:
            transition_counts[current_state, next_state] += 1
            
        # Normalize to probabilities
        # Add a small epsilon to avoid division by zero if a state is never visited (Laplace smoothing optional, here we just check row sums)
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        # Handle states with no outgoing transitions (though unlikely in this dataset)
        row_sums[row_sums == 0] = 1 
        
        self.transition_matrix = transition_counts / row_sums
        print("Dataset fitted successfully.")
        
    def predict_next_state_probs(self, current_val):
        """Returns the probability distribution for the next state given a current continuous value."""
        if self.transition_matrix is None:
            raise ValueError("Model is not fitted yet.")
            
        # Determine current state
        # Hand-rolling digitize for single value efficiency/clarity
        # pd.cut logic:
        current_state = pd.cut([current_val], bins=self.bins, labels=range(self.num_bins), include_lowest=True)[0]
        
        return self.transition_matrix[current_state]

    def predict_next_expected_value(self, current_val):
        """Predicts the expected precipitation probability for the next time step."""
        probs = self.predict_next_state_probs(current_val)
        # Expected value is sum(prob * state_center_value)
        return np.sum(probs * self.state_centers)

    def evaluate(self, df):
        """Calculates RMSE on the provided dataset."""
        print("Evaluating model...")
        # Ensure generation of transitions respects region boundaries
        df = df.sort_values(by=['region', 'time'])
        predictions = []
        actuals = []
        
        for _, group in df.groupby('region'):
            vals = group['precipitation_probability (%)'].values
            # Predict next value from current value
            # Inputs: vals[:-1], Targets: vals[1:]
            if len(vals) < 2:
                continue
            
            for i in range(len(vals) - 1):
                curr = vals[i]
                target = vals[i+1]
                pred = self.predict_next_expected_value(curr)
                predictions.append(pred)
                actuals.append(target)
                
        if not predictions:
            return 0.0
            
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
        return rmse

def main():
    dataset_path = r'dataset/train.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Basic Check
    required_cols = ['time', 'region', 'precipitation_probability (%)']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}'. Available columns: {df.columns}")
            return

    # Preprocessing
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time for 80/20 sequential split
    df = df.sort_values(by='time')
    
    # 80/20 Train/Val Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Initialize and Fit Model
    markov_model = MarkovPrecipitationModel(num_bins=10) # 10 bins: 0-10, 10-20, ...
    markov_model.fit(train_df)

    print("\nModel Training Complete.")
    print("-" * 30)
    print("Transition Matrix Shape:", markov_model.transition_matrix.shape)
    
    # Evaluate
    rmse = markov_model.evaluate(val_df)
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Demonstration
    print("\n--- Prediction Demonstration ---")
    test_values = [0, 5, 15, 50, 85, 95]
    print(f"{'Current Precip %':<20} | {'Predicted Next (Expected %)':<30} | {'Most Likely Next State (Bin Mid)':<30}")
    print("-" * 80)
    
    for val in test_values:
        expected_next = markov_model.predict_next_expected_value(val)
        probs = markov_model.predict_next_state_probs(val)
        most_likely_state_idx = np.argmax(probs)
        most_likely_val = markov_model.state_centers[most_likely_state_idx]
        
        print(f"{val:<20} | {expected_next:<30.2f} | {most_likely_val:<30.1f}")

    # Optional: Visualize a Transition
    print("\nTransition probabilities from State 0 (0-10%):")
    print(markov_model.transition_matrix[0])

    # --- Test Prediction & Submission ---
    print("\n--- Generating Submission ---")
    test_path = r'dataset/test.csv'
    if not os.path.exists(test_path):
        print(f"Error: Test dataset not found at {test_path}")
        return

    try:
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error reading Test CSV: {e}")
        return

    # Retrain on FULL dataset for best performance
    print("Retraining on full training dataset...")
    markov_model.fit(df) 

    # Prepare Test Data
    # sorting is crucial for autoregressive prediction
    test_df['time'] = pd.to_datetime(test_df['time'])
    test_df = test_df.sort_values(by=['region', 'time'])
    
    # Get last known value from training data for each region to start the chain
    # We assume the test set immediately follows the training set chronologically
    last_train_values = df.sort_values('time').groupby('region')['precipitation_probability (%)'].last()
    
    predictions = []
    
    # Iterate by region in test set
    for region, group in test_df.groupby('region'):
        if region not in last_train_values.index:
            # Fallback if region not in train (unlikely) -> start with mean or 0
            current_val = df['precipitation_probability (%)'].mean()
            print(f"Warning: Region {region} not found in train data. Using mean: {current_val:.2f}")
        else:
            current_val = last_train_values[region]
            
        region_preds = []
        for _ in range(len(group)):
            # Predict next step
            next_val = markov_model.predict_next_expected_value(current_val)
            region_preds.append(next_val)
            # Update current_val for autoregression
            current_val = next_val
            
        # Add to the list (group is already sorted, so order matches iteration)
        # However, we need to be careful to map back to original indices if we want to submit
        # But here we append to a list that corresponds to the sorted test_df
        predictions.extend(region_preds)

    # Assign predictions back to sorted test_df
    test_df['precipitation_probability (%)'] = predictions
    
    # Create submission file
    # Ensure we include all rows and required columns, matching sample_submission format if needed.
    # Usually submission is ID, target.
    submission = test_df[['ID', 'precipitation_probability (%)']].copy()
    
    # Sort back by ID if necessary, or just keep as is if ID order doesn't matter (usually it doesn't, but good practice)
    # Let's just write what we have.
    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission.head())

if __name__ == "__main__":
    main()
