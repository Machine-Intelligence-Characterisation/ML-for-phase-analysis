import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

# TODO: EVERYTHING IN HERE SUCKS

def load_results(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def safe_f1_score(y_true, y_pred, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f1_score(y_true, y_pred, zero_division=0, **kwargs)

def analyze_results(results):
    for task in results['predictions'].keys():
        predictions = np.array(results['predictions'][task])
        labels = np.array(results['labels'][task])
        
        print(f"\nAnalysis for task: {task}")
        print(f"Shape of predictions: {predictions.shape}")
        print(f"Shape of labels: {labels.shape}")
        
        if task == 'composition':
            # Convert single-label predictions to multi-label format
            mlb = MultiLabelBinarizer(classes=range(labels.shape[1]))
            predictions_multilabel = mlb.fit_transform([[pred] for pred in predictions])
            
            print("Multi-label Classification Metrics:")
            print(f"Subset Accuracy: {accuracy_score(labels, predictions_multilabel):.4f}")
            print(f"Micro F1 Score: {safe_f1_score(labels, predictions_multilabel, average='micro'):.4f}")
            print(f"Macro F1 Score: {safe_f1_score(labels, predictions_multilabel, average='macro'):.4f}")
            print(f"Weighted F1 Score: {safe_f1_score(labels, predictions_multilabel, average='weighted'):.4f}")
            
            print("\nPer-element Performance:")
            for i in range(labels.shape[1]):
                element_f1 = safe_f1_score(labels[:, i], predictions_multilabel[:, i])
                if element_f1 > 0:  # Only print for elements that appear in the dataset
                    print(f"Element {i}: F1 Score = {element_f1:.4f}")
            
            print("\nMost common misclassifications:")
            misclassifications = []
            for true, pred in zip(labels, predictions_multilabel):
                true_elements = set(np.where(true == 1)[0])
                pred_elements = set(np.where(pred == 1)[0])
                false_positives = pred_elements - true_elements
                false_negatives = true_elements - pred_elements
                for elem in false_positives:
                    misclassifications.append((elem, 0, 1))
                for elem in false_negatives:
                    misclassifications.append((elem, 1, 0))
            
            if misclassifications:
                misclass_counts = {}
                for elem, true, pred in misclassifications:
                    key = (elem, true, pred)
                    misclass_counts[key] = misclass_counts.get(key, 0) + 1
                
                sorted_misclass = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)
                for (elem, true, pred), count in sorted_misclass[:5]:  # Top 5 misclassifications
                    print(f"Element {elem}: True = {true}, Predicted = {pred}, Count = {count}")
        else:
            # Single-label classification analysis
            print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
            
            print("\nClassification Report:")
            print(classification_report(labels, predictions, zero_division=0))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(labels, predictions))
        
            print("\nMost common misclassifications:")
            misclassifications = [(label, pred) for label, pred in zip(labels, predictions) if label != pred]
            
            if misclassifications:
                misclass_counts = {}
                for true, pred in misclassifications:
                    key = (true, pred)
                    misclass_counts[key] = misclass_counts.get(key, 0) + 1
                
                sorted_misclass = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)
                for (true, pred), count in sorted_misclass[:5]:  # Top 5 misclassifications
                    print(f"True: {true}, Predicted: {pred}, Count: {count}")
            else:
                print("No misclassifications found!")

def main():
    file_path = 'analysis/inference_analysis/inference_data/TEST.pkl'
    results = load_results(file_path)
    analyze_results(results)

if __name__ == "__main__":
    main()