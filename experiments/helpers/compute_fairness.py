
import os, sys, dotenv

dotenv.load_dotenv()
root_path = os.environ.get("SCRIPT_PATH")
sys.path.append(root_path)
from data.metadata import feat_dict
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric



def calculate_all_fairness_metrics(model, df_test, dataset_name):
    """
    Calculate fairness metrics for all sensitive attributes in a dataset.
    
    Parameters:
        model: Trained model with a predict method
        df_test: Test dataframe
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary of fairness metrics for each sensitive attribute
    """
    X_test = df_test.drop(feat_dict[dataset_name]['target'], axis=1)
    preds = model.predict(X_test.values)
    df_test = df_test.copy()
    df_test['predicted_label'] = preds
    
    label_name = feat_dict[dataset_name]['target']
    
    # Set favorable/unfavorable labels based on the dataset
    favorable_label = 0.0 if dataset_name == 'compas' else 1.0
    unfavorable_label = 1.0 if dataset_name == 'compas' else 0.0
    
    metrics = {}
    
    # Calculate metrics for each sensitive attribute
    for sens_attr in feat_dict[dataset_name]['sens_attrs']:
        print(f"Calculating fairness metrics for {sens_attr} in {dataset_name}")
        
        # Create BinaryLabelDataset for true values
        true_ds = BinaryLabelDataset(
            df=df_test,
            label_names=[label_name],
            protected_attribute_names=[sens_attr],
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label
        )
        
        # Create BinaryLabelDataset for predicted values
        pred_ds = true_ds.copy(deepcopy=True)
        pred_ds.labels = df_test['predicted_label'].values.reshape(-1, 1)
        
        # Set privileged/unprivileged groups
        if ((dataset_name == 'compas') and (sens_attr=='sex')):
            privileged_groups = [{sens_attr: 0}]
            unprivileged_groups = [{sens_attr: 1}]
        else:
            privileged_groups = [{sens_attr: 1}]
            unprivileged_groups = [{sens_attr: 0}]
        
        # Calculate fairness metrics
        metric = ClassificationMetric(
            true_ds, pred_ds,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups
        )
        
        # Store metrics
        metrics[sens_attr] = {
            'statistical_parity_difference': metric.statistical_parity_difference(),
            'disparate_impact': metric.disparate_impact(),
            'average_odds_difference': metric.average_odds_difference(),
            'equal_opportunity_difference': metric.equal_opportunity_difference()
        }
    
    return metrics


