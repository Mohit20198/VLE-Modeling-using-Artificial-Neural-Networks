# ANN Surrogate Model for Azeotropic Vapor-Liquid Equilibrium Prediction

This repository contains the implementation of an Artificial Neural Network (ANN) surrogate model for predicting Vapor-Liquid Equilibrium (VLE) in the Ethanol-Water binary system at 101.325 kPa. The model is designed to capture the non-ideal, azeotropic behavior of this system, providing a fast and accurate alternative to traditional thermodynamic models. 

## Abstract

Accurate prediction of Vapor-Liquid Equilibrium (VLE) is critical for designing separation processes but is computationally expensive using traditional thermodynamic models. This project develops an ANN as a rapid surrogate model for the Ethanol-Water system at 101.325 kPa. A thermodynamically consistent dataset of 500 VLE points was generated using the Wilson and Antoine equations. The machine learning workflow includes:

- **Data Preprocessing**: 80-20 train-test split, feature normalization with `MinMaxScaler`.
- **Feature Engineering**: Inclusion of the activity coefficient (γ₁) as a physically informative feature.
- **Hyperparameter Tuning**: 5-fold cross-validation to optimize ANN architecture, learning rate, and batch size.
- **Physics-Informed Training**: Weighted loss to prioritize the azeotropic region and L2 regularization to prevent overfitting.
- **Evaluation**: Comparison against Raoult's Law baseline.

The final ANN achieved an RMSE of 0.03192 on the test set, significantly outperforming Raoult's Law (RMSE = 0.06903), and predicted the azeotrope at `x₁ = 0.9390` (reference: `x₁ = 0.894`), demonstrating a balance between statistical accuracy and physical realism.

**Keywords**: Surrogate Modeling, Artificial Neural Network, Vapor-Liquid Equilibrium, Azeotrope, Ethanol-Water, Machine Learning, Chemical Engineering.

## Project Structure

- **`vle_data_generation.py`**: Generates a thermodynamically consistent dataset using Antoine and Wilson equations.
- **`ann_model_training.py`**: Implements the ANN model with preprocessing, hyperparameter tuning, physics-informed loss, and evaluation.
- **`generated_dataset/`**: Directory containing the generated VLE dataset (`vle_data.csv`).
- **`figures/`**: Directory for output plots (e.g., parity plot, y-x diagram).
- **`README.md`**: This file, providing project overview and instructions.

## Methodology

### Data Generation
- **Dataset**: 500 VLE points for Ethanol-Water at 101.325 kPa.
- **Models**:
  - **Antoine Equation**: Calculates saturation pressure (`Pᵢˢᵃᵗ`) for ethanol and water.
  - **Wilson Model**: Computes activity coefficients (`γ₁`, `γ₂`) for non-ideal liquid phase behavior.
  - **VLE Equation**: Solves for bubble point temperature (`T`) and vapor mole fraction (`y₁`).
- **Azeotrope Focus**: Higher data density in the azeotropic region (`x₁ ≈ 0.85–0.95`).

### ANN Model
- **Architecture**: Two hidden dense layers with ReLU activation, L2 regularization (0.001), and a sigmoid output layer for `y₁ ∈ [0, 1]`.
- **Preprocessing**: Input features (`x₁`, `T`, `P`, `γ₁`) normalized using `MinMaxScaler`.
- **Hyperparameter Tuning**: Grid search over layer sizes, learning rates, and batch sizes using 5-fold cross-validation.
- **Physics-Informed Loss**: Weighted loss (20x for azeotropic region) and bubble point consistency penalty.
- **Training**: Implemented in TensorFlow with a custom training loop to incorporate physics constraints.

### Evaluation
- **Metrics**: RMSE and MAE on the test set.
- **Baseline**: Raoult's Law, which assumes ideal behavior and fails to capture azeotrope.
- **Azeotrope Detection**: Physics-informed approach using a fine grid to predict `x₁` where `y₁ = x₁`.

## Results

### Performance
| Model                  | RMSE    | MAE     |
|------------------------|---------|---------|
| ANN Model              | 0.03192 | 0.02306 |
| Raoult's Law (Baseline)| 0.06903 | 0.04823 |

- The ANN significantly outperforms Raoult's Law, capturing the non-ideal VLE behavior.
- Azeotrope predicted at `x₁ = 0.9390` (reference: `x₁ = 0.894`), with an absolute error of 0.045, indicating good physical realism.

### Visualizations
- **Parity Plot**: High correlation between predicted and actual `y₁` on the test set.
- **y-x Diagram**: Correctly captures the azeotropic curve shape and identifies the azeotrope.
- **Training Loss**: Shows convergence of MSE and physics-informed loss components.
- **Residual Plot**: Analyzes prediction errors across `x₁`.

## Installation and Usage

### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy pandas scipy tensorflow scikit-learn matplotlib
  ```

### Running the Code
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/vle-ann-surrogate.git
   cd vle-ann-surrogate
   ```

2. **Generate Dataset**:
   ```bash
   python vle_data_generation.py
   ```
   This creates `generated_dataset/vle_data.csv`.

3. **Train and Evaluate Model**:
   ```bash
   python ann_model_training.py
   ```
   This performs preprocessing, hyperparameter tuning, training with physics-informed loss, and generates evaluation plots.

4. **Outputs**:
   - Dataset: `generated_dataset/vle_data.csv`
   - Plots: Saved in `figures/` (e.g., parity plot, y-x diagram)
   - Console: Prints RMSE, MAE, azeotrope predictions, and timing.

### Example Output
```
Best Hyperparameters: Layers=(128, 64), LR=0.005, Batch=16
ANN RMSE: 0.03192, MAE: 0.02306
Raoult's Law RMSE: 0.06903, MAE: 0.04823
Predicted Azeotrope: x₁ = 0.9390, y₁ = 0.9390, T = 351.52 K
Reference Azeotrope: x₁ = 0.8940, T = 351.30 K
```

## Contributing
Contributions are welcome! Please submit issues or pull requests for:
- Improving azeotrope prediction accuracy (e.g., adjusting physics loss weight).
- Adding new features (e.g., other VLE systems or models).
- Optimizing computational efficiency.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## References
1. J. M. Smith, H. C. Van Ness, M. M. Abbott, *Introduction to Chemical Engineering Thermodynamics*, 8th ed., McGraw-Hill, 2018.
2. G. M. Wilson, "Vapor-Liquid Equilibrium. XI. A New Expression for the Excess Free Energy of Mixing," *Journal of the American Chemical Society*, vol. 86, no. 2, pp. 127–130, 1964.
3. K. Hornik, M. Stinchcombe, H. White, "Multilayer feedforward networks are universal approximators," *Neural Networks*, vol. 2, no. 5, pp. 359-366, 1989.
4. M. Abadi et al., "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems," tensorflow.org, 2015.
5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

## Contact
For questions or feedback, contact Mohit Upadhyay at [your.email@example.com].
