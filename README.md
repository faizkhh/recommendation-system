# recommendation-system
A movie recommendation system using the MovieLens 100K dataset.  
Implemented User-Based CF, Item-Based CF, and SVD-based Matrix Factorization 

---

## 🚀 Features
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization using **SVD**
- Cosine Similarity & Pearson Correlation
- Sparse User–Item Matrix handling
- Optimized for low-end systems

---

## 📊 Dataset
### Dataset

Download MovieLens 100K dataset from [here](https://grouplens.org/datasets/movielens/100k/)

Place the `u.data` and `u.item` files inside a `data/` folder in the project root.

- **MovieLens 100K**
- 100,000 ratings
- 943 users
- 1,682 movies

---

## 🛠 Tech Stack
- Python
- NumPy
- Pandas

---

## 📂 Project Structure
recommendation-system/
│
├── data/
│   ├── u.data
│   ├── u.item
│
├── src/
│   ├── similarity.py
│   ├── user_based_cf.py
│   ├── item_based_cf.py
│   ├── matrix_builder.py
│   └── data_loader.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

