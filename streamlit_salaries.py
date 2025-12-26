import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =======================
# ЗАГРУЗКА МОДЕЛЕЙ
# =======================
reg_model = joblib.load("salary_reg_model.pkl")
clf_model = joblib.load("salary_clf_model.pkl")

# =======================
# ЗАГОЛОВОК
# =======================
st.title("💼 Data Science Salary Predictor")
st.markdown(
    """
    **Регрессия** — прогноз зарплаты (USD)  
    **Классификация** — уровень зарплаты (low / mid / high)
    """
)

st.markdown("### Введите параметры специалиста")

# =======================
# ВВОД ДАННЫХ
# =======================
experience_level = st.selectbox(
    "Уровень опыта",
    ["EN", "MI", "SE", "EX"]
)

company_size = st.selectbox(
    "Размер компании",
    ["S", "M", "L"]
)

remote_ratio = st.selectbox(
    "Формат работы",
    {
        "On-site (0%)": 0,
        "Hybrid (50%)": 50,
        "Remote (100%)": 100
    }.keys()
)

company_location = st.selectbox(
    "Локация компании",
    ["US", "Non-US"]
)

work_year = st.selectbox(
    "Год",
    [2021, 2022, 2023]
)

# =======================
# ПОДГОТОВКА ДАННЫХ (БЕЗ КОДИРОВАНИЯ!)
# =======================
input_df = pd.DataFrame([{
    "work_year": work_year,
    "remote_ratio": {
        "On-site (0%)": 0,
        "Hybrid (50%)": 50,
        "Remote (100%)": 100
    }[remote_ratio],
    "experience_level": experience_level,
    "company_location": "US" if company_location == "US" else "GB",
    "company_size": company_size
}])

# =======================
# ПРЕДСКАЗАНИЕ
# =======================
# =======================
# ПРЕДСКАЗАНИЕ
# =======================
if st.button("🔮 Предсказать"):

    # --- Регрессия ---
    log_salary_pred = reg_model.predict(input_df)[0]
    salary_usd = np.expm1(log_salary_pred)

    # --- Классификация ---
    salary_class_num = clf_model.predict(input_df)[0]

    salary_class_map = {
        0: "low",
        1: "mid",
        2: "high"
    }

    salary_class = salary_class_map[int(salary_class_num)]

    # --- Вывод ---
    st.success(f"Ожидаемая зарплата: **${salary_usd:,.0f} USD**")
    st.info(
        f"🏷 Уровень зарплаты: **{salary_class.upper()}**"
    )
