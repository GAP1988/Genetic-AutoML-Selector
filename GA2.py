import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'Student_Performance.csv'
data = pd.read_csv(file_path)

# Display columns to verify
print("Columns in the dataset:")
print(data.columns)

# Define target column name
target_column = 'Performance Index'  # Replace with the actual name of the target column
# Define an imputer for numerical columns (e.g., using mean)
numerical_imputer = SimpleImputer(strategy='mean')

# Convert target column to numeric
data[target_column] = pd.to_numeric(data[target_column], errors='coerce')

# Separate features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Drop rows with NaN target values
X = X[~y.isna()]
y = y.dropna()

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', numerical_imputer), ('scaler', StandardScaler())]), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])


# Exploratory Data Analysis (EDA)
# Histograms for numerical features
# X[numerical_columns].hist(bins=15, figsize=(15, 6), layout=(2, 3))
#plt.tight_layout()
#plt.show()
# Number of numerical columns
num_numerical_columns = len(numerical_columns)

# Calculate appropriate layout
rows = (num_numerical_columns // 3) + (num_numerical_columns % 3 > 0)
columns = 3

X[numerical_columns].hist(bins=15, figsize=(15, 6), layout=(rows, columns))
plt.tight_layout()
plt.show()

# Bar plots for categorical features
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=X, x=col)
    plt.title(f'Distribution of {col}')
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data[numerical_columns + [target_column]].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Define a ColumnTransformer for preprocessing
#preprocessor = ColumnTransformer(
 #   transformers=[
  #      ('num', StandardScaler(), numerical_columns),
   #     ('cat', OneHotEncoder(), categorical_columns)
    #])

# Apply the transformations
X_processed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define model types and hyperparameter ranges
MODELS = {
    0: ('SVR', SVR),
    1: ('RandomForestRegressor', RandomForestRegressor),
    2: ('KNeighborsRegressor', KNeighborsRegressor),
    3: ('DecisionTreeRegressor', DecisionTreeRegressor)
}

# Genetic algorithm parameters
POP_SIZE = 10
NGEN = 10
CXPB = 0.5
MUTPB = 0.2

# Initialize population
def initialize_population(pop_size, ind_size):
    return [[random.random() for _ in range(ind_size)] for _ in range(pop_size)]

# Evaluate an individual
def evaluate(individual):
    model_type_idx = int(individual[0] * len(MODELS))
    model_type, model_class = MODELS[model_type_idx]
    params = {}

    if model_type == 'SVR':
        params['C'] = individual[1] * 10  # Example parameter range
        params['gamma'] = individual[2] * 0.1
    elif model_type == 'RandomForestRegressor':
        params['n_estimators'] = int(individual[1] * 100)
        params['max_depth'] = int(individual[2] * 10) + 1  # Ensure min value is 1
    elif model_type == 'KNeighborsRegressor':
        params['n_neighbors'] = int(individual[1] * 20) + 1  # Ensure min value is 1
        params['leaf_size'] = int(individual[2] * 50) + 1  # Ensure min value is 1
    elif model_type == 'DecisionTreeRegressor':
        params['max_depth'] = int(individual[1] * 20) + 1  # Ensure min value is 1
        params['min_samples_split'] = int(individual[2] * 10) + 2  # Ensure min value is 2

    model = model_class(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return -mse  # We minimize MSE, so we return negative MSE as fitness

# Selection
def select(population, fitnesses, k=3):
    selected = []
    for _ in range(len(population)):
        aspirants = [random.choice(population) for _ in range(k)]
        aspirant_fitnesses = [fitnesses[population.index(aspirant)] for aspirant in aspirants]
        selected.append(aspirants[np.argmax(aspirant_fitnesses)])
    return selected

# Crossover
def crossover(parent1, parent2, cxpb):
    if random.random() < cxpb:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

# Mutation
def mutate(individual, mutpb):
    for i in range(len(individual)):
        if random.random() < mutpb:
            individual[i] = random.random()
    return individual

# Genetic algorithm
population = initialize_population(POP_SIZE, 3)
best_individuals = []
best_fitnesses = []

for gen in range(NGEN):
    fitnesses = [evaluate(ind) for ind in population]
    selected = select(population, fitnesses)
    offspring = []
    for i in range(0, len(selected), 2):
        if i + 1 < len(selected):
            child1, child2 = crossover(selected[i], selected[i + 1], CXPB)
            offspring.extend([child1, child2])
        else:
            offspring.append(selected[i])
    population = [mutate(ind, MUTPB) for ind in offspring]

    # Track best individual
    best_individual = max(population, key=evaluate)
    best_fitness = evaluate(best_individual)
    best_individuals.append(best_individual)
    best_fitnesses.append(best_fitness)

    print(f"Generation {gen}: Best fitness: {-best_fitness}")

# Evaluate the final best individual
best_ind = max(population, key=evaluate)
best_fitness = evaluate(best_ind)

print(f"Best individual is {best_ind}, with fitness: {-best_fitness}")

# Plotting the best fitness over generations
plt.figure(figsize=(10, 5))
plt.plot(range(NGEN), [-f for f in best_fitnesses], marker='o')
plt.title('Best Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

# Extracting model details for the best individual
model_type_idx = int(best_ind[0] * len(MODELS))
model_type, model_class = MODELS[model_type_idx]
params = {}

if model_type == 'SVR':
    params['C'] = best_ind[1] * 10  # Example parameter range
    params['gamma'] = best_ind[2] * 0.1
elif model_type == 'RandomForestRegressor':
    params['n_estimators'] = int(best_ind[1] * 100)
    params['max_depth'] = int(best_ind[2] * 10) + 1  # Ensure min value is 1
elif model_type == 'KNeighborsRegressor':
    params['n_neighbors'] = int(best_ind[1] * 20) + 1  # Ensure min value is 1
    params['leaf_size'] = int(best_ind[2] * 50) + 1  # Ensure min value is 1
elif model_type == 'DecisionTreeRegressor':
    params['max_depth'] = int(best_ind[1] * 20) + 1  # Ensure min value is 1
    params['min_samples_split'] = int(best_ind[2] * 10) + 2  # Ensure min value is 2

best_model = model_class(**params)
best_model.fit(X_train, y_train)
best_predictions = best_model.predict(X_test)
best_mse = mean_squared_error(y_test, best_predictions)

print(f"Best model details: {model_type} with parameters: {params}")
print(f"Mean Squared Error of the best model on the test set: {best_mse}")
