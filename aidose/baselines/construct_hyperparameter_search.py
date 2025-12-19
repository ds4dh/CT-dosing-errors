

def construct_hyperparameter_search(param, trial, scale_pos_weight) -> dict:
    """"
    Construct the hyperparameter search space used by Optuna for optimizing XGBoost hyperaparameters.
    The search space is defined based on the label (classification or regression) and the model type.
    """
    if param.model == 'XGBoost' or param.model == 'LateFusionModel':
        return _construct_xgboost_hyperparameter_search(param, trial, scale_pos_weight)
    else:
        raise NotImplementedError(f"There is not implemented method to construct the hyperparameter search for {param.model}.")


def _construct_xgboost_hyperparameter_search(param, trial, scale_pos_weight) -> dict:
    """
    Construct the hyperparameter search space for XGBoost model.
    """

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": param.random_seed,
    }

    # specific to the binary task
    if param.label == 'wilson_label':
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "aucpr"
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.5 * scale_pos_weight, 2.0 * scale_pos_weight)

    # specific to the regression task
    if param.label in ['dosing_error_rate', 'sum_dosing_errors']:
        sample_alpha = trial.suggest_float("sample_alpha", 5.0, 200.0, log=True)
        params["objective"] = trial.suggest_categorical("objective", ["reg:absoluteerror", "reg:pseudohubererror"])

    return params