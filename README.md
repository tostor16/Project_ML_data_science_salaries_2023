# Data Science Salary Predictor

Проект посвящён анализу зарплат специалистов в области Data Science и построению моделей машинного обучения для их прогнозирования.

Реализованы две задачи:
- регрессия — прогноз заработной платы в USD
- классификация — определение уровня зарплаты (low / mid / high)

Проект выполнен на основе реального датасета с вакансиями Data Science специалистов.

Ссылка на дата сет: https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023/data

Исходные признаки дата сета:
work_year: The year the salary was paid.
experience_level: The experience level in the job during the year
employment_type: The type of employment for the role
job_title: The role worked in during the year.
salary: The total gross salary amount paid.
salary_currency: The currency of the salary paid as an ISO 4217 currency code.
salaryinusd: The salary in USD
employee_residence: Employee's primary country of residence in during the work year as an ISO 3166 country code.
remote_ratio: The overall amount of work done remotely
company_location: The country of the employer's main office or contracting branch
company_size: The median number of people that worked for the company during the year

---

## Структура проекта
├── DataScienceSalary.ipynb # Анализ данных и обучение моделей
├── app.py # Streamlit-приложение
├── ridge_salary_model.joblib # Сохранённая регрессионная модель
├── salary_classification_model.joblib # Сохранённая модель классификации
├── requirements.txt
└── README.md

В качестве признаков используются:
- уровень опыта (experience_level)
- размер компании (company_size)
- формат работы (remote_ratio)
- локация компании (company_location)
- год (work_year)

Целевая переменная для регрессии:
- логарифм зарплаты (log_salary)

Для классификации зарплата разбита на три класса с помощью квантилей:
- low
- mid
- high

## Модели

### Регрессия
- Linear Regression
- Ridge Regression
- Lasso Regression

Лучшая модель:
- Ridge Regression

Метрики качества:
- RMSE
- MAE
- R²

### Классификация
- Logistic Regression
- Random Forest

Метрики качества:
- Accuracy
- Precision
- Recall
- F1-score

## Streamlit-приложение

Реализован пользовательский интерфейс, позволяющий:
- вводить параметры специалиста
- получать прогноз зарплаты в USD
- определять уровень зарплаты (low / mid / high)

Модели используются в виде сохранённых пайплайнов (`joblib`), что гарантирует корректную обработку данных.

## Используемые технологии

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib, Seaborn
