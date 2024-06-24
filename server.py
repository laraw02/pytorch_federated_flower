# Credits. This code has been adapted from :
# https://github.com/adap/flower/tree/main/examples/advanced-tensorflow

from typing import Dict, Optional, Tuple, List
import flwr as fl

#all packages to save data
import os
import numpy as np
import pandas as pd


save_evaluation = True
save_updated_model = True

# server address = {IP_ADDRESS}:{PORT}
server_address = "10.46.134.6:5050"

federatedLearningcounts = 3

file_name = "loss_metrics"

def main() -> None:
    # load and compile model for : server-side parameter initialization, server-side parameter evaluation
    
    # loading and compiling Keras model, choose either MobileNetV2 (faster) or EfficientNetB0. 
    # feel free to add more Keras applications
    # https://keras.io/api/applications/
    """
    Model               MobileNetV2     EfficientNetB0
    Size (MB)           14              29
    Top-1 Accuracy      71.3%           77.1%	
    Top-5 Accuracy      90.1%           93.3%
    Parameters          3.5M            5.3M
    Depth               105             132	
    CPU inference ms    25.9            46.0
    GPU inference ms    3.8             4.9
    """

    """
    # uncomment to load an EfficientNetB0 model
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(160, 160, 3), weights=None, classes=2
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    """
    
    
    class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
        def aggregate_evaluate(
            self,
            server_round,
            results,
            failures,
        ):
            """Aggregate evaluation accuracy using weighted average and write results to .csv file"""

            if not results:
                return None, {}

            # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            # Aggregate and print custom metric
            aggregated_accuracy = sum(accuracies) / sum(examples)
            #print(f"AGGREGATION OF CLIENT MODELS EVALUATED")

            if save_evaluation: 
                new_data = {'aggregated_loss': [aggregated_loss], 'aggregated_accuracy': [aggregated_accuracy]}
                df = pd.DataFrame(new_data)

            # Check if the file exists
                if os.path.isfile(file_name):
                # Append new data to the existing CSV file
                    df.to_csv(file_name, mode='a', header=False, index=False)
                else:
                # If the file does not exist, write the header as well
                    df.to_csv(file_name, mode='w', header=True, index=False)
                
                # Return aggregated loss and metrics (i.e., aggregated accuracy)
            return aggregated_loss, {"accuracy": aggregated_accuracy}


        def aggregate_fit(
        self,
        server_round,
        results,
        failures,
        ):
            """Save updated global model after each federeated learning round"""

            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if save_updated_model:

                if aggregated_parameters is not None:
                # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Save aggregated_ndarrays
                    print(f"Saving round {server_round} aggregated_ndarrays...")
                    np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

            return aggregated_parameters, aggregated_metrics

            

        # create strategy
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    # start Flower server (SSL-enabled) for X rounds of federated learning
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=federatedLearningcounts),
        strategy=strategy
    )



if __name__ == "__main__":
    main()
