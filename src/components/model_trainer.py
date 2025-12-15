import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor   
) 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            logging.info("Starting model evaluation...")
            model_report: dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                  models=models)
            
            print("\n" + "="*50)
            print("MODEL EVALUATION RESULTS:")
            print("="*50)
            for model_name, score in sorted(model_report.items(), key=lambda x: x[1], reverse=True):
                print(f"{model_name:.<30} R² = {score:.4f}")
            print("="*50 + "\n")
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R² >= 0.6")
            
            logging.info(f"Best model found: {best_model_name} with R² = {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    import numpy as np
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    print("="*50)
    print("RUNNING COMPLETE ML PIPELINE")
    print("="*50)
    
    try:
        
        print("\n Step 1: Data Ingestion...")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print(f"✅ Train data: {train_path}")
        print(f"✅ Test data: {test_path}")
        
        
        print("\n🔄 Step 2: Data Transformation...")
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )
        print(f"✅ Train array shape: {train_arr.shape}")
        print(f"✅ Test array shape: {test_arr.shape}")
        print(f"✅ Preprocessor saved: {preprocessor_path}")
        
       
        print("\n Step 3: Model Training...")
        trainer = ModelTrainer()
        r2_score = trainer.initiate_model_trainer(train_arr, test_arr)
        
        print("\n" + "="*50)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Best Model R² Score: {r2_score:.4f}")
        print(f"Model saved at: {trainer.model_trainer_config.trained_model_file_path}")
        
    except Exception as e:
        print(f"\n ERROR occurred:")
        print(e)
        import traceback
        traceback.print_exc()