# 16-11-25

### 1. **Scalar (x)**
- **Definition:** A scalar is simply a single number. It represents a one-dimensional quantity.
- **Example:**
  - \( x = 5 \)
  - \( x = -3.14 \)
- **Use:** Scalars are used for quantities like temperature, mass, or any single value.

---

### 2. **Vector (x)**
- **Definition:** A vector is an ordered list of numbers. It can be thought of as a one-dimensional array. Vectors are often used to represent quantities that have both magnitude and direction, like velocity or force.
- **Notation:**
  - \( \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \)
- **Example:**
  - \( \mathbf{x} = \begin{bmatrix} 2 \\ 5 \\ 7 \end{bmatrix} \) is a 3-dimensional vector.
- **Element:** \( x_i \) refers to the \( i \)-th element of the vector \( \mathbf{x} \).

---

### 3. **Matrix (X)**
- **Definition:** A matrix is a two-dimensional array of numbers arranged in rows and columns. It is often used to represent linear transformations, systems of equations, or datasets.
- **Notation:**
  - \( \mathbf{X} = \begin{bmatrix}
    x_{11} & x_{12} & \cdots & x_{1m} \\
    x_{21} & x_{22} & \cdots & x_{2m} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{n1} & x_{n2} & \cdots & x_{nm}
    \end{bmatrix} \)
- **Example:**
  - \( \mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \) is a 2x2 matrix.
- **Element:** \( x_{ij} \) or \( [\mathbf{X}]_{ij} \) refers to the element at the \( i \)-th row and \( j \)-th column of matrix \( \mathbf{X} \).

---

### 4. **General Tensor (X)**
- **Definition:** A tensor is a generalization of scalars, vectors, and matrices to higher dimensions. A scalar is a 0D tensor, a vector is a 1D tensor, and a matrix is a 2D tensor. Tensors can have three or more dimensions.
- **Example:**
  - A 3D tensor could represent a cube of numbers, like a series of matrices stacked together.
- **Use:** Tensors are widely used in fields like physics (e.g., stress tensors) and machine learning (e.g., data batches in neural networks).

---

### 5. **Identity Matrix (I)**
- **Definition:** The identity matrix is a special square matrix with ones on the main diagonal and zeros everywhere else. It acts like the number 1 in matrix multiplication: multiplying any matrix by the identity matrix leaves the original matrix unchanged.
- **Notation:**
  - \( \mathbf{I} = \begin{bmatrix}
    1 & 0 & \cdots & 0 \\
    0 & 1 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & 1
    \end{bmatrix} \)
- **Example:**
  - \( \mathbf{I} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \) is a 2x2 identity matrix.

---

### 6. **Element of a Vector (\( x_i \), \( [\mathbf{x}]_i \))**
- **Definition:** This notation refers to the \( i \)-th element of a vector \( \mathbf{x} \).
- **Example:**
  - For \( \mathbf{x} = \begin{bmatrix} 2 \\ 5 \\ 7 \end{bmatrix} \), \( x_2 = 5 \).

---

### 7. **Element of a Matrix (\( x_{ij} \), \( [\mathbf{X}]_{ij} \))**
- **Definition:** This notation refers to the element of a matrix \( \mathbf{X} \) located at the \( i \)-th row and \( j \)-th column.
- **Example:**
  - For \( \mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \), \( x_{21} = 3 \).

---

### Summary Table


Summary of Numerical Objects


| Object         | Definition                                                                 | Example                                                                 |
|----------------|----------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Scalar         | Single number                                                             | \( x = 5 \)                                                             |
| Vector         | Ordered list of numbers (1D array)                                        | \( \mathbf{x} = \begin{bmatrix} 2 \\ 5 \\ 7 \end{bmatrix} \)         |
| Matrix         | 2D array of numbers                                                       | \( \mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \)      |
| General Tensor | Generalization to higher dimensions                                       | 3D tensor: cube of numbers                                               |
| Identity Matrix| Square matrix with 1s on the diagonal and 0s elsewhere                  | \( \mathbf{I} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \)      |
| Vector Element | \( i \)-th element of vector \( \mathbf{x} \)                           | \( x_2 = 5 \)                                                            |
| Matrix Element | Element at row \( i \), column \( j \) of matrix \( \mathbf{X} \)      | \( x_{21} = 3 \)                                                        |

