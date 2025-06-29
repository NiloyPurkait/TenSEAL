# Enabling Bootstrapping for CKKSVector in TenSEAL: Step-by-Step Notes

This document explains, step by step, how bootstrapping support was added to TenSEAL's CKKSVector, which files were changed, and how to use the new feature in practice.

---

## 1. C++ Backend: CKKSVector Bootstrapping Implementation

**Files changed:**
- `tenseal/cpp/tensors/ckksvector.h`
- `tenseal/cpp/tensors/ckksvector.cpp`

**What was added:**
- Declaration and implementation of the `void bootstrap();` method for the `CKKSVector` class.
- The method loops over all ciphertexts in the vector and calls SEAL's `bootstrap_inplace` on each, refreshing the ciphertext and reducing noise.

---

## 2. Python Bindings (pybind11)

**File changed:**
- `tenseal/binding.cpp`

**What was added:**
- Exposed the new `bootstrap()` method to Python by adding a pybind11 binding for `CKKSVector::bootstrap()`.

---

## 3. Python API

**File changed:**
- `tenseal/tensors/ckksvector.py`

**What was added:**
- Added a `bootstrap(self)` method to the `CKKSVector` Python class, which calls the underlying C++ method.
- This allows users to call `bootstrap()` on any CKKSVector object in Python.

---

## 4. Unit Test

**File changed:**
- `tests/python/tenseal/tensors/test_ckks_vector.py`

**What was added:**
- A test function `test_bootstrap_ckksvector()` that:
  - Creates a CKKSVector, performs several multiplications to increase noise,
  - Calls `bootstrap()` to refresh the ciphertext,
  - Decrypts and checks that the result is still accurate.

---

## 5. Jupyter Notebook Usage Example

**File changed:**
- `tutorials/experimental/FairEncrypt.ipynb`

**What was added:**
- Markdown and code cells demonstrating how to use `bootstrap()` on encrypted vectors:
  - After encryption (to enable deeper computation on test data)
  - After the encrypted forward pass (to allow further encrypted operations)
  - Comparison of accuracy and runtime with and without bootstrapping
- Explanations of when and why to use bootstrapping in privacy-preserving ML workflows.

---

## How Bootstrapping Works in TenSEAL

- **Purpose:** Bootstrapping refreshes a CKKS ciphertext, reducing accumulated noise and enabling further encrypted computation (e.g., more layers, more multiplications).
- **How to use:** In Python, simply call `.bootstrap()` on any CKKSVector:
  ```python
  enc_vec.bootstrap()
  ```
- **When to use:** After several homomorphic operations, or before running additional encrypted layers or polynomial activations. Bootstrapping is computationally expensive, so use it only when necessary.

---

## Summary Table

| File                                         | Purpose                                  |
|----------------------------------------------|------------------------------------------|
| `tenseal/cpp/tensors/ckksvector.h/.cpp`      | C++ backend: implement `bootstrap()`     |
| `tenseal/binding.cpp`                        | pybind11: expose `bootstrap()` to Python |
| `tenseal/tensors/ckksvector.py`              | Python API: add `bootstrap()` method     |
| `tests/python/tenseal/tensors/test_ckks_vector.py` | Unit test for bootstrapping        |
| `tutorials/experimental/FairEncrypt.ipynb`   | Usage and documentation in notebook      |

---

## Next Steps
- You can now use bootstrapping in your encrypted ML workflows by calling `.bootstrap()` on any CKKSVector.
- See the notebook for practical examples and scientific context.
