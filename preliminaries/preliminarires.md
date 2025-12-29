# Preliminaries

Resume at: 2.5. Automatic Differentiation

## 2.3.10. Matrix–Matrix Multiplication

### **1. What are A and B?**

- **A** is a 2×3 matrix:
  ```
  [[0., 1., 2.],
   [3., 4., 5.]]
  ```
- **B** is a 3×4 matrix:
  ```
  [[1., 1., 1., 1.],
   [1., 1., 1., 1.],
   [1., 1., 1., 1.]]
  ```

---

### **2. Matrix Multiplication Rules**

For matrix multiplication (`A@B`), the number of **columns in A** must match the number of **rows in B**.
- Here, A is 2×3 and B is 3×4, so the result will be a **2×4** matrix.

The element at row `i`, column `j` of the result is the **dot product** of the `i`-th row of A and the `j`-th column of B.

---

### **3. Calculating the Result**

Let's compute each element of the resulting 2×4 matrix:

#### **First row of result (A[0] @ B):**
- **Result[0,0] = (0×1) + (1×1) + (2×1) = 0 + 1 + 2 = 3**
- **Result[0,1] = (0×1) + (1×1) + (2×1) = 0 + 1 + 2 = 3**
- **Result[0,2] = (0×1) + (1×1) + (2×1) = 0 + 1 + 2 = 3**
- **Result[0,3] = (0×1) + (1×1) + (2×1) = 0 + 1 + 2 = 3**

#### **Second row of result (A[1] @ B):**
- **Result[1,0] = (3×1) + (4×1) + (5×1) = 3 + 4 + 5 = 12**
- **Result[1,1] = (3×1) + (4×1) + (5×1) = 3 + 4 + 5 = 12**
- **Result[1,2] = (3×1) + (4×1) + (5×1) = 3 + 4 + 5 = 12**
- **Result[1,3] = (3×1) + (4×1) + (5×1) = 3 + 4 + 5 = 12**

---

### **4. Final Result**

So, the result of `A@B` is:
```
[[ 3.,  3.,  3.,  3.],
 [12., 12., 12., 12.]]
```

---

