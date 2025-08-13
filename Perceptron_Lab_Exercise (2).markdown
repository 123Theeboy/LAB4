# Perceptron Lab Exercise: Sorting Books with a Robot Librarian

## Introduction
Welcome to the Perceptron Lab! In this exercise, you’ll explore the **Perceptron**, a simple machine learning model that classifies data into two categories, like sorting books into **fiction (+1)** or **non-fiction (-1)**. Think of the Perceptron as a **robot librarian** who learns to sort books based on features like size and color. You’ll run code, visualize how the librarian learns, and experiment with different settings to understand how the Perceptron works.

### Analogy: The Robot Librarian
- **Setup (`__init__`)**: The librarian chooses how fast to learn (`eta`) and how many times to practice (`n_iter`).
- **Training (`fit`)**: They look at books (data), guess based on features (e.g., size, color), and adjust their guessing rules (weights) if they make mistakes.
- **Prediction (`net_input`, `predict`)**: For a new book, they calculate a score and decide if it’s fiction or non-fiction.
- **Docstring**: A guidebook explaining the librarian’s tools and results.

## Goals
- Run the Perceptron code to classify a simple dataset of books.
- Visualize the learning process and decision boundary.
- Experiment with the librarian’s learning speed (`eta`) and practice rounds (`n_iter`).
- Answer questions to understand how the Perceptron learns and predicts.

## Prerequisites
- Basic Python knowledge (lists, loops, functions).
- Install Python libraries: `numpy`, `matplotlib` (run `pip install numpy matplotlib` in a terminal).
- A Python environment (e.g., Jupyter Notebook, VS Code, or an IDE like PyCharm).

## Lab Setup
Below is the Perceptron code you’ll use. It implements a binary classifier that learns to separate two classes by adjusting weights.

```python
import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data."""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
```

## Exercise 1: Running the Perceptron
You’ll use a simple dataset to train the Perceptron and predict whether a new book is fiction (+1) or non-fiction (-1).

### Dataset
The dataset represents 3 books with features `[size, color]`:
- `X = np.array([[2, 3], [1, 1], [4, 5]])` (size and color for 3 books).
- `y = np.array([1, -1, 1])` (fiction=+1, non-fiction=-1).
- Interpretation:
  - Book 1: `[size=2, color=3]` → fiction (+1)
  - Book 2: `[size=1, color=1]` → non-fiction (-1)
  - Book 3: `[size=4, color=5]` → fiction (+1)

### Task
1. Copy the Perceptron code into a Python environment (e.g., a Jupyter Notebook).
2. Run the following code to train the Perceptron and predict for a new book:

```python
import numpy as np
X = np.array([[2, 3], [1, 1], [4, 5]])  # Features: size, color
y = np.array([1, -1, 1])  # Labels: fiction (+1), non-fiction (-1)
model = Perceptron(eta=0.1, n_iter=10)
model.fit(X, y)
print("Prediction for new book [3, 2]:", model.predict(np.array([3, 2])))
print("Errors per epoch:", model.errors_)
```

3. **Record the output**. You should see something like:
   ```
   Prediction for new book [3, 2]: -1
   Errors per epoch: [2, 1, 2, 1, 1, 1, 0, 0, 0, 0]
   ```

### Questions
1. What does the prediction `-1` mean for the book `[size=3, color=2]`? (Hint: Look at the labels in `y`.)
2. How many total errors did the Perceptron make across all 10 epochs? (Sum the numbers in `errors_`.)
3. Why do the errors drop to 0 by epoch 7? What does this tell you about the dataset?

## Exercise 2: Visualizing Learning Progress
The `errors_` list shows how many books the librarian got wrong in each epoch. Let’s plot this to see how the librarian improves over time.

### Task
1. Add the following code after running Exercise 1 to plot the errors:

```python
import matplotlib.pyplot as plt
plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of Errors')
plt.title('Perceptron Learning Progress')
plt.grid(True)
plt.show()
```

2. Run the code and observe the plot.
3. **Record observations**:
   - How do the errors change over the 10 epochs?
   - In which epoch does the librarian stop making mistakes?

### Questions
1. Why do the errors go up and down (e.g., 2, 1, 2, 1) before reaching 0?
2. What does it mean when the errors reach 0? (Hint: Think about the librarian’s sorting rule and the data.)

## Exercise 3: Visualizing the Decision Boundary
The Perceptron learns a line in 2D space (size vs. color) to separate fiction (+1) from non-fiction (-1) books. Let’s visualize this line and see where the new book `[3, 2]` falls.

### Task
1. Add the following code after Exercise 1 to plot the data points and decision boundary:

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot data points
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='Fiction (+1)')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='Non-fiction (-1)')
plt.scatter([3], [2], color='green', marker='*', s=200, label='New book [3, 2]')

# Plot decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdBu')
plt.xlabel('Size')
plt.ylabel('Color')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()
```

2. Run the code and observe the plot:
   - **Blue circles**: Fiction books (`+1`).
   - **Red crosses**: Non-fiction books (`-1`).
   - **Green star**: New book `[3, 2]`.
   - **Shaded regions**: Blue for fiction (+1), red for non-fiction (-1).

3. **Record observations**:
   - Where is the new book `[3, 2]` relative to the decision boundary?
   - Does the boundary separate the fiction and non-fiction books correctly?

### Questions
1. Why does the new book `[3, 2]` get a prediction of `-1`? (Hint: Look at its position relative to the boundary.)
2. How does the decision boundary (line) separate the fiction and non-fiction books?
3. If you test a new book at `[4, 4]`, what prediction would you expect? Why? (Hint: Check which side of the boundary it falls on.)

## Exercise 4: Experimenting with Learning Parameters
The librarian’s learning speed (`eta`) and number of practice rounds (`n_iter`) affect how they learn to sort books. Let’s experiment with these settings.

### Task
1. Modify the Perceptron parameters in Exercise 1 and rerun the code with:
   - **Setting 1**: `eta=0.01` (slower learning), `n_iter=20`.
   - **Setting 2**: `eta=0.5` (faster learning), `n_iter=5`.
2. For each setting, record:
   - The prediction for `[3, 2]`.
   - The errors list (`model.errors_`).
3. Plot the errors for each setting using the code from Exercise 2.

### Questions
1. How does changing `eta` (learning rate) affect the errors list? Does slower (`0.01`) or faster (`0.5`) learning make the errors drop faster?
2. How does changing `n_iter` (epochs) affect the results? Did fewer epochs (`n_iter=5`) still reach 0 errors?
3. Did the prediction for `[3, 2]` change with different settings? Why or why not? (Hint: Think about whether the final decision boundary changes.)

## Exercise 5: Exploring a Real-World Dataset (Iris)
Let’s apply the Perceptron to a real-world dataset: the **Iris dataset**, classifying two types of flowers (Setosa vs. Versicolor) based on petal length and width.

### Task
1. Install scikit-learn: `pip install scikit-learn`.
2. Run the following code to load the Iris dataset, train the Perceptron, and make a prediction:

```python
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
X = iris.data[:100, [2, 3]]  # Petal length, petal width (Setosa and Versicolor)
y = iris.target[:100]  # 0 for Setosa, 1 for Versicolor
y = np.where(y == 0, -1, 1)  # Convert to -1 (Setosa) and 1 (Versicolor)
model = Perceptron(eta=0.1, n_iter=10)
model.fit(X, y)
print("Prediction for new flower [4.0, 1.0]:", model.predict(np.array([[4.0, 1.0]])))
print("Errors per epoch:", model.errors_)
```

3. Plot the errors (use Exercise 2 code) and decision boundary (adapt Exercise 3 code, replacing `size` with `petal length` and `color` with `petal width`).
4. **Record**:
   - The prediction for `[4.0, 1.0]`.
   - The errors list.
   - Observations from the plots.

### Questions
1. What does the prediction for `[4.0, 1.0]` mean? Is it Setosa (-1) or Versicolor (+1)?
2. Does the errors list reach 0? Why or why not? (Hint: Research whether Setosa and Versicolor are linearly separable.)
3. How does the decision boundary for the Iris data compare to the book dataset? Is it easier or harder to separate the classes?

## Bonus Challenge: Modify the Dataset
Let’s see how the librarian handles a new book in the original dataset.

### Task
1. Add a new book to the dataset: `[3, 4]` with label `+1` (fiction). Update the data:
   ```python
   X = np.array([[2, 3], [1, 1], [4, 5], [3, 4]])
   y = np.array([1, -1, 1, 1])
   ```
2. Retrain the Perceptron (use `eta=0.1`, `n_iter=10`) and record:
   - The prediction for `[3, 2]`.
   - The errors list.
3. Plot the decision boundary and errors.

### Questions
1. Did the prediction for `[3, 2]` change after adding the new book? Why?
2. How did the errors list change? Did it still reach 0? (Hint: Is the new dataset still linearly separable?)
3. Try changing `random_state` (e.g., 42 or 100). Does it affect the prediction or errors?

## Teaching Tips for Students
- **Visualize**: The plots show how the librarian sorts books. The decision boundary plot (Exercise 3) shows a line separating fiction (blue) from non-fiction (red), with the new book as a green star.
- **Simplify Math**: The `net_input` method is like calculating a score for a book: `score = size * weight_size + color * weight_color + bias`. The `predict` method checks if the score is positive (fiction, +1) or negative (non-fiction, -1).
- **Show Progress**: The errors plot (Exercise 2) shows the librarian making fewer mistakes as they learn, like getting better at sorting with practice.
- **Hands-On**: Experimenting with `eta`, `n_iter`, and new data helps you see how the librarian’s learning changes.

## Submission
- Submit your Python code, plots (save as images), and answers to all questions.
- Write a short paragraph (3–5 sentences) explaining what you learned about the Perceptron using the robot librarian analogy.

## Conclusion
In this lab, you’ve acted as a robot librarian, using the Perceptron to sort books (or flowers) into two categories. You’ve seen how it learns a decision boundary, visualized its progress, and experimented with settings. The Perceptron is a simple but powerful model, and it’s the foundation of modern neural networks!