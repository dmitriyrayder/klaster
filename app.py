import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet не встановлено. Прогнози будуть недоступні.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from io import BytesIO

st.set_page_config(page_title="Аналіз товарів", layout="wide")

# Ініціалізація session_state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None
if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = None

st.title("🔍 Аналіз товарів: визначення кандидатів на зняття")

# === СТИЛІЗАЦІЯ ===
st.markdown("""
<style>
/* Анімації */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 10px rgba(99, 102, 241, 0.5); }
    50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.8); }
}

/* Основний контейнер з градієнтом */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}

/* Головний блок контенту */
.main .block-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    animation: fadeIn 0.8s ease-out;
}

/* Заголовки з градієнтами */
h1, h2, h3 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    padding: 0.5rem 0;
    border-bottom: 3px solid;
    border-image: linear-gradient(90deg, #667eea, #764ba2, #f093fb) 1;
    margin-bottom: 1.5rem;
    animation: fadeIn 1s ease-out;
}

h1 {
    font-size: 2.5rem !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

h2 {
    font-size: 2rem !important;
}

h3 {
    font-size: 1.5rem !important;
}

/* Sidebar стилізація */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.2);
}

section[data-testid="stSidebar"] .block-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: white !important;
    -webkit-text-fill-color: white !important;
    border-bottom: none !important;
}

/* Кнопки з градієнтами та анімацією */
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
    cursor: pointer;
}

.stButton button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
}

.stButton button:active {
    transform: translateY(-1px);
}

/* Primary кнопка */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    animation: glow 2s ease-in-out infinite;
}

.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
}

/* Текстові поля з ефектом свічення */
.stTextInput input,
.stSelectbox select,
.stTextArea textarea {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 0.75rem;
    transition: all 0.3s ease;
    background: white;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.stTextInput input:focus,
.stSelectbox select:focus,
.stTextArea textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2),
                0 4px 12px rgba(102, 126, 234, 0.3);
    outline: none;
}

/* File uploader */
.stFileUploader {
    border: 3px dashed #667eea;
    border-radius: 15px;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(240, 147, 251, 0.05) 100%);
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: #764ba2;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%);
    transform: scale(1.02);
}

/* Metrics (метрики) */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #667eea;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    animation: fadeIn 1s ease-out;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

div[data-testid="stMetric"] label {
    font-weight: 600;
    color: #667eea;
}

/* Таблиці (DataFrames) */
.stDataFrame {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    animation: fadeIn 1s ease-out;
}

.dataframe {
    border: none !important;
}

/* Заголовки таблиць з градієнтом */
.dataframe thead tr th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem !important;
    border: none !important;
    text-align: left !important;
}

/* Чергування рядків таблиці */
.dataframe tbody tr:nth-child(even) {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(240, 147, 251, 0.05) 100%) !important;
}

.dataframe tbody tr:nth-child(odd) {
    background: white !important;
}

.dataframe tbody tr:hover {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(240, 147, 251, 0.15) 100%) !important;
    transform: scale(1.01);
    transition: all 0.2s ease;
}

.dataframe tbody td {
    padding: 0.75rem !important;
    border-bottom: 1px solid #f0f0f0 !important;
}

/* Info, Warning, Success, Error блоки */
.stAlert {
    border-radius: 12px;
    border-left: 5px solid;
    padding: 1rem 1.5rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    animation: fadeIn 0.6s ease-out;
}

div[data-baseweb="notification"] {
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

/* Info блок */
.stAlert[data-baseweb="notification"][kind="info"],
div[data-baseweb="notification"]:has(> div > svg[data-testid="stInfoIcon"]) {
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
    border-left-color: #4facfe;
}

/* Success блок */
.stAlert[kind="success"],
div[data-baseweb="notification"]:has(> div > svg[data-testid="stSuccessIcon"]) {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, rgba(34, 197, 94, 0.1) 100%);
    border-left-color: #34d399;
}

/* Warning блок */
.stAlert[kind="warning"],
div[data-baseweb="notification"]:has(> div > svg[data-testid="stWarningIcon"]) {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%);
    border-left-color: #fbbf24;
}

/* Error блок */
.stAlert[kind="error"],
div[data-baseweb="notification"]:has(> div > svg[data-testid="stErrorIcon"]) {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
    border-left-color: #ef4444;
}

/* Expander (розгортаємі блоки) */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%);
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    padding: 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(240, 147, 251, 0.2) 100%);
    border-color: #667eea;
    transform: translateX(5px);
}

.streamlit-expanderContent {
    border: 2px solid #e0e0e0;
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 1rem;
    background: white;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}

/* Слайдери */
.stSlider {
    padding: 1rem 0;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Radio buttons */
div[role="radiogroup"] label {
    background: white;
    padding: 0.75rem 1.5rem;
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    margin: 0.25rem;
    transition: all 0.3s ease;
    cursor: pointer;
}

div[role="radiogroup"] label:hover {
    border-color: #667eea;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%);
    transform: scale(1.05);
}

div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: #667eea;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

/* Колонки з рамками */
div[data-testid="column"] {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid #f0f0f0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    animation: fadeIn 0.8s ease-out;
}

div[data-testid="column"]:hover {
    border-color: #667eea;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    transform: translateY(-3px);
}

/* Горизонтальний розділювач */
hr {
    border: none;
    height: 3px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    margin: 2rem 0;
    border-radius: 2px;
    animation: fadeIn 1s ease-out;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 10px;
}

/* Checkbox */
.stCheckbox {
    padding: 0.5rem;
}

/* Spinner */
div[data-testid="stSpinner"] > div {
    border-top-color: #667eea !important;
}

/* Download button */
.stDownloadButton button {
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    box-shadow: 0 6px 20px rgba(52, 211, 153, 0.4);
    transition: all 0.3s ease;
}

.stDownloadButton button:hover {
    background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(52, 211, 153, 0.6);
}

/* Scrollbar стилізація */
::-webkit-scrollbar {
    width: 12px;
    height: 12px;
}

::-webkit-scrollbar-track {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    transition: all 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Caption (підпис) */
.caption {
    color: #6b7280;
    font-style: italic;
    text-align: center;
    padding: 1rem;
    border-top: 2px solid #f0f0f0;
    margin-top: 2rem;
}

/* Графіки в карточках */
div[data-testid="stPlotlyChart"],
div[data-testid="stVegaLiteChart"],
div[data-testid="stPyplot"] {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid #f0f0f0;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    transition: all 0.3s ease;
    animation: fadeIn 1s ease-out;
}

div[data-testid="stPlotlyChart"]:hover,
div[data-testid="stVegaLiteChart"]:hover,
div[data-testid="stPyplot"]:hover {
    border-color: #667eea;
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    transform: translateY(-5px);
}

/* Мобільна адаптація */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
        border-radius: 10px;
    }

    h1 {
        font-size: 1.8rem !important;
    }

    h2 {
        font-size: 1.5rem !important;
    }

    h3 {
        font-size: 1.2rem !important;
    }

    div[data-testid="column"] {
        padding: 1rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        padding: 1rem;
    }

    .stButton button {
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
}

/* Додаткові ефекти для таблиць результатів */
.stDataFrame div[role="gridcell"] {
    transition: all 0.2s ease;
}

/* Виділення важливих значень */
.stDataFrame div[role="gridcell"]:has(> div:contains("🚫")) {
    font-weight: 600;
    color: #ef4444;
}

.stDataFrame div[role="gridcell"]:has(> div:contains("⚠️")) {
    font-weight: 600;
    color: #fbbf24;
}

.stDataFrame div[role="gridcell"]:has(> div:contains("✅")) {
    font-weight: 600;
    color: #34d399;
}

/* Плавна поява контенту при скролі */
@media (prefers-reduced-motion: no-preference) {
    * {
        scroll-behavior: smooth;
    }
}
</style>
""", unsafe_allow_html=True)

# === НАЛАШТУВАННЯ ===
with st.sidebar:
    st.header("⚙️ Налаштування")
    TOP_N = st.slider("Кількість топ-артикулів для Prophet", 10, 50, 20)

    st.subheader("🎯 Критерії зняття")
    zero_weeks_threshold = st.slider("Тижнів підряд без продажів", 8, 20, 12)
    min_total_sales = st.slider("Мінімальний обсяг продажів", 1, 50, 5)
    max_store_ratio = st.slider("Макс. частка магазинів без продажів (%)", 70, 95, 85, 5) / 100

    st.subheader("🤖 Модель ML")
    use_balanced_model = st.checkbox("Використовувати балансування класів", value=True)
    final_threshold = st.slider("Фінальний поріг для зняття (%)", 50, 90, 70, 5) / 100

    st.divider()

    # Кнопка очищення кешу
    if st.button("🔄 Очистити кеш даних"):
        st.session_state.loaded_data = None
        st.cache_data.clear()
        st.success("Кеш очищено!")
        st.rerun()

# === ЗАВАНТАЖЕННЯ ДАНИХ ===
st.header("📁 Завантаження даних")
st.info("💡 Формат: дата, артикул, кількість, магазин, назва")

# Вибір джерела даних
data_source = st.radio(
    "Оберіть джерело даних:",
    ["Google Sheets", "Локальний файл"],
    horizontal=True
)

uploaded_file = None
sheets_url = None

if data_source == "Локальний файл":
    uploaded_file = st.file_uploader("Оберіть Excel файл", type=['xlsx', 'xls'])
else:
    sheets_url = st.text_input(
        "Посилання на Google Sheets:",
        value="https://docs.google.com/spreadsheets/d/1lJLON5N_EKQ5ICv0Pprp5DamP1tNAhBIph4uEoWC04Q/edit?gid=64159818#gid=64159818",
        help="Таблиця повинна мати публічний доступ"
    )

# === КЕШОВАНІ ФУНКЦІЇ ЗАВАНТАЖЕННЯ ===
@st.cache_data(show_spinner=False)
def _fetch_google_sheets_data(sheets_url):
    """Кешоване завантаження сирих даних з Google Sheets"""
    import re
    import time

    # Витягуємо spreadsheet ID
    spreadsheet_match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheets_url)
    if not spreadsheet_match:
        raise ValueError("Невірний формат посилання на Google Sheets")

    spreadsheet_id = spreadsheet_match.group(1)

    # Витягуємо GID (ID аркуша)
    gid_match = re.search(r'[#&]gid=([0-9]+)', sheets_url)
    gid = gid_match.group(1) if gid_match else '0'

    # Формуємо URL для експорту в Excel форматі
    export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=xlsx&gid={gid}"

    # Завантажуємо дані з прогрес-баром
    progress_bar = st.progress(0, text="🔄 Підключення до Google Sheets...")
    time.sleep(0.3)
    progress_bar.progress(20, text="📥 Завантаження даних...")

    df = pd.read_excel(export_url, nrows=100000)

    progress_bar.progress(80, text="✅ Обробка даних...")
    time.sleep(0.2)
    progress_bar.progress(100, text="✅ Завантаження завершено!")
    time.sleep(0.3)
    progress_bar.empty()

    return df

@st.cache_data(show_spinner=False)
def _load_excel_file(file_bytes, sheet_name):
    """Кешоване завантаження Excel файлу"""
    from io import BytesIO
    import time

    progress_bar = st.progress(0, text="📂 Відкриття файлу...")
    time.sleep(0.2)
    progress_bar.progress(30, text="📊 Читання даних...")

    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, nrows=100000)

    progress_bar.progress(90, text="✅ Фіналізація...")
    time.sleep(0.2)
    progress_bar.progress(100, text="✅ Файл завантажено!")
    time.sleep(0.3)
    progress_bar.empty()

    return df

def load_and_process_data(uploaded_file):
    if uploaded_file is None:
        st.info("👆 Завантажте Excel файл для початку роботи")
        return None, False

    try:
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)

        if file_size > 50 * 1024 * 1024:
            st.error("❌ Файл занадто великий. Максимум: 50MB")
            return None, False

        # Визначаємо аркуші
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        excel_file = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Оберіть аркуш:", excel_file.sheet_names) if len(excel_file.sheet_names) > 1 else excel_file.sheet_names[0]

        # Використовуємо кешоване завантаження
        df = _load_excel_file(file_bytes, selected_sheet)
        if len(df) == 100000:
            st.warning("⚠️ Файл обрізано до 100,000 рядків")

        st.success(f"✅ Завантажено {len(df)} рядків")
        
        # Співставлення колонок
        available_cols = list(df.columns)
        col1, col2 = st.columns(2)

        with col1:
            date_col = st.selectbox("Дата:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['дат', 'date'])), 0))
            art_col = st.selectbox("Артикул:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['арт', 'art'])), 0))
            qty_col = st.selectbox("Кількість:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['кол', 'кіл', 'qty', 'кількість', 'количество'])), 0))

        with col2:
            magazin_col = st.selectbox("Магазин:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['маг', 'magazin', 'магазин'])), 0))
            name_col = st.selectbox("Назва:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['назв', 'name', 'назва', 'название'])), 0))
            segment_col = st.selectbox("Сегмент (опціонально):", ['Без сегментації'] + available_cols)
        
        # Перейменування колонок
        column_mapping = {date_col: 'Data', art_col: 'Art', qty_col: 'Qty', magazin_col: 'Magazin', name_col: 'Name'}
        if segment_col != 'Без сегментації':
            column_mapping[segment_col] = 'Segment'

        df = df.rename(columns=column_mapping)

        # Перевірка обов'язкових колонок
        required_cols = ['Data', 'Art', 'Qty', 'Magazin', 'Name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Відсутні колонки: {missing_cols}")
            return None, False

        # Фільтрація по сегменту
        if 'Segment' in df.columns:
            st.subheader("🎯 Вибір сегмента")
            unique_segments = sorted(df['Segment'].dropna().unique())
            selected_segment = st.selectbox("Сегмент:", ['Всі сегменти'] + list(unique_segments))

            if selected_segment != 'Всі сегменти':
                df = df[df['Segment'] == selected_segment].copy()
                st.success(f"✅ Обрано сегмент: {selected_segment}")

        with st.expander("📊 Попередній перегляд"):
            st.dataframe(df.head())
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Записів", len(df))
            with col2: st.metric("Артикулів", df['Art'].nunique())
            with col3:
                try:
                    date_min = pd.to_datetime(df['Data'], errors='coerce').min()
                    date_max = pd.to_datetime(df['Data'], errors='coerce').max()
                    st.metric("Період", f"{date_min.strftime('%Y-%m-%d')} - {date_max.strftime('%Y-%m-%d')}")
                except:
                    st.metric("Період", "Помилка дат")

        return df, True

    except Exception as e:
        st.error(f"❌ Помилка завантаження: {str(e)}")
        return None, False

def load_from_google_sheets(sheets_url):
    """Завантаження даних з публічної Google Sheets таблиці"""
    if not sheets_url or sheets_url.strip() == "":
        st.info("👆 Введіть посилання на Google Sheets")
        return None, False

    try:
        # Використовуємо кешоване завантаження даних
        df = _fetch_google_sheets_data(sheets_url)

        if len(df) == 100000:
            st.warning("⚠️ Файл обрізано до 100,000 рядків")

        st.success(f"✅ Завантажено {len(df)} рядків з Google Sheets")

        # Співставлення колонок (ідентично load_and_process_data)
        available_cols = list(df.columns)
        col1, col2 = st.columns(2)

        with col1:
            date_col = st.selectbox("Дата:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['дат', 'date'])), 0), key="gs_date")
            art_col = st.selectbox("Артикул:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['арт', 'art'])), 0), key="gs_art")
            qty_col = st.selectbox("Кількість:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['кол', 'кіл', 'qty', 'кількість', 'количество'])), 0), key="gs_qty")

        with col2:
            magazin_col = st.selectbox("Магазин:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['маг', 'magazin', 'магазин'])), 0), key="gs_magazin")
            name_col = st.selectbox("Назва:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['назв', 'name', 'назва', 'название'])), 0), key="gs_name")
            segment_col = st.selectbox("Сегмент (опціонально):", ['Без сегментації'] + available_cols, key="gs_segment")

        # Перейменування колонок
        column_mapping = {date_col: 'Data', art_col: 'Art', qty_col: 'Qty', magazin_col: 'Magazin', name_col: 'Name'}
        if segment_col != 'Без сегментації':
            column_mapping[segment_col] = 'Segment'

        df = df.rename(columns=column_mapping)

        # Перевірка обов'язкових колонок
        required_cols = ['Data', 'Art', 'Qty', 'Magazin', 'Name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Відсутні колонки: {missing_cols}")
            return None, False

        # Фільтрація по сегменту
        if 'Segment' in df.columns:
            st.subheader("🎯 Вибір сегмента")
            unique_segments = sorted(df['Segment'].dropna().unique())
            selected_segment = st.selectbox("Сегмент:", ['Всі сегменти'] + list(unique_segments), key="gs_segment_filter")

            if selected_segment != 'Всі сегменти':
                df = df[df['Segment'] == selected_segment].copy()
                st.success(f"✅ Обрано сегмент: {selected_segment}")

        with st.expander("📊 Попередній перегляд"):
            st.dataframe(df.head())
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Записів", len(df))
            with col2: st.metric("Артикулів", df['Art'].nunique())
            with col3:
                try:
                    date_min = pd.to_datetime(df['Data'], errors='coerce').min()
                    date_max = pd.to_datetime(df['Data'], errors='coerce').max()
                    st.metric("Період", f"{date_min.strftime('%Y-%m-%d')} - {date_max.strftime('%Y-%m-%d')}")
                except:
                    st.metric("Період", "Помилка дат")

        return df, True

    except Exception as e:
        st.error(f"❌ Помилка завантаження з Google Sheets: {str(e)}")
        st.info("💡 Переконайтеся, що таблиця має публічний доступ")
        return None, False

# Завантаження даних в залежності від обраного джерела з використанням session_state
# Перевіряємо, чи змінилось джерело даних
if st.session_state.data_source_type != data_source:
    st.session_state.loaded_data = None  # Скидаємо кеш при зміні джерела
    st.session_state.data_source_type = data_source

# Якщо дані вже завантажені і джерело не змінилось, використовуємо кешовані
if st.session_state.loaded_data is not None:
    df, data_loaded = st.session_state.loaded_data
    if data_loaded:
        st.info("ℹ️ Використовуються раніше завантажені дані")
else:
    # Завантажуємо нові дані
    if data_source == "Локальний файл":
        df, data_loaded = load_and_process_data(uploaded_file)
    else:
        df, data_loaded = load_from_google_sheets(sheets_url)

    # Зберігаємо в session_state
    if data_loaded:
        st.session_state.loaded_data = (df, data_loaded)

if data_loaded:
    st.header("🚀 Запуск аналізу")
    if st.button("▶️ ПОЧАТИ АНАЛІЗ", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

    if not st.session_state.get('run_analysis', False):
        st.info("👆 Натисніть кнопку для запуску аналізу")
        st.stop()
else:
    st.stop()

# === ОСНОВНА ОБРОБКА ===
def process_data(df):
    with st.spinner("🔄 Обробка даних..."):
        # Очищення даних
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Data'])
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
        df = df[df['Qty'] >= 0]

        if len(df) == 0:
            st.error("❌ Немає валідних даних")
            st.stop()

        df['year_week'] = df['Data'].dt.strftime('%Y-%U')

        # Обмеження артикулів
        all_arts = df['Art'].unique()
        if len(all_arts) > 5000:
            st.warning("⚠️ Обробляємо топ-5000 артикулів за продажами")
            top_arts = df.groupby('Art')['Qty'].sum().nlargest(5000).index
            all_arts = top_arts
            df = df[df['Art'].isin(all_arts)]

        # Агрегація по тижнях
        weekly = df.groupby(['Art', 'year_week'])['Qty'].sum().reset_index()
        unique_weeks = sorted(df['year_week'].unique())
        all_weeks = pd.MultiIndex.from_product([all_arts, unique_weeks], names=['Art', 'year_week'])
        weekly = weekly.set_index(['Art', 'year_week']).reindex(all_weeks, fill_value=0).reset_index()

        return df, weekly, all_arts, unique_weeks

def calculate_abc_xyz_analysis(df):
    # ABC аналіз
    abc_analysis = df.groupby('Art').agg({
        'Qty': ['sum', 'mean', 'std'],
        'Data': ['min', 'max']
    }).reset_index()
    
    abc_analysis.columns = ['Art', 'total_qty', 'avg_qty', 'std_qty', 'first_sale', 'last_sale']
    abc_analysis['days_in_catalog'] = (abc_analysis['last_sale'] - abc_analysis['first_sale']).dt.days + 1

    # ABC категорії (виправлено: сортування перед кумулятивним розрахунком)
    abc_analysis = abc_analysis.sort_values('total_qty', ascending=False).reset_index(drop=True)
    total_sum = abc_analysis['total_qty'].sum()

    # Захист від ділення на нуль
    if total_sum > 0:
        abc_analysis['cum_qty'] = abc_analysis['total_qty'].cumsum()
        abc_analysis['cum_qty_pct'] = abc_analysis['cum_qty'] / total_sum
    else:
        abc_analysis['cum_qty'] = 0
        abc_analysis['cum_qty_pct'] = 0

    def get_abc_category(cum_pct):
        if cum_pct <= 0.8: return 'A'
        elif cum_pct <= 0.95: return 'B'
        else: return 'C'

    abc_analysis['abc_category'] = abc_analysis['cum_qty_pct'].apply(get_abc_category)

    # XYZ аналіз (виправлено: обробка нульових значень)
    abc_analysis['coefficient_variation'] = np.where(
        abc_analysis['avg_qty'] > 0,
        abc_analysis['std_qty'] / abc_analysis['avg_qty'],
        999  # Велике значення для товарів без продажів
    )

    def get_xyz_category(cv):
        if cv <= 0.1: return 'X'  # Стабільний попит
        elif cv <= 0.25: return 'Y'  # Помірно мінливий
        else: return 'Z'  # Нестабільний попит

    abc_analysis['xyz_category'] = abc_analysis['coefficient_variation'].apply(get_xyz_category)

    return abc_analysis

def calculate_features(weekly, df):
    def compute_features(group):
        sorted_group = group.sort_values('year_week')
        qty_series = sorted_group['Qty'].values
        
        if len(qty_series) == 0:
            return pd.Series({
                'ma_3': 0, 
                'ma_6': 0, 
                'consecutive_zeros': 0,
                'zero_weeks_12': 0, 
                'trend': 0
            })


        # Ковзні середні
        qty_series_pd = pd.Series(qty_series)
        ma_3 = qty_series_pd.rolling(3, min_periods=1).mean().iloc[-1]
        ma_6 = qty_series_pd.rolling(6, min_periods=1).mean().iloc[-1]

        # Послідовні нулі з кінця
        consecutive_zeros = 0
        for val in reversed(qty_series):
            if val == 0:
                consecutive_zeros += 1
            else:
                break

        # Нулі за останні 12 тижнів
        zero_weeks_12 = int(np.sum(qty_series[-12:] == 0)) if len(qty_series) >= 12 else int(np.sum(qty_series == 0))

        # Тренд
        trend = 0
        if len(qty_series) >= 4:
            try:
                x = np.arange(len(qty_series))
                coeffs = np.polyfit(x, qty_series, 1)
                trend = float(coeffs[0])
            except:
                trend = 0
        
        return pd.Series({
            'ma_3': float(ma_3), 
            'ma_6': float(ma_6), 
            'consecutive_zeros': int(consecutive_zeros),
            'zero_weeks_12': int(zero_weeks_12), 
            'trend': float(trend)
        })
    
    # Застосовуємо функцію і отримуємо DataFrame з Art в індексі
    features = weekly.groupby('Art').apply(compute_features, include_groups=False).reset_index()

    # Розрахунок частки магазинів без продажів
    total_stores = df['Magazin'].nunique()

    if total_stores == 0:
        st.error("❌ Не знайдено магазинів в даних")
        st.stop()

    # Магазини з продажами для кожного артикула
    stores_with_sales = df[df['Qty'] > 0].groupby('Art')['Magazin'].nunique().reset_index()
    stores_with_sales.columns = ['Art', 'stores_with_sales']
    stores_with_sales['no_store_ratio'] = 1 - (stores_with_sales['stores_with_sales'] / total_stores)

    features = features.merge(stores_with_sales[['Art', 'no_store_ratio']], on='Art', how='left')
    features['no_store_ratio'] = features['no_store_ratio'].fillna(1.0)

    return features

def create_ml_model(features, abc_analysis):
    # Створення міток для навчання (ВИПРАВЛЕНА ЛОГІКА)
    def create_labels(row):
        score = 0

        # Категорія C - агресивні критерії
        if row['abc_category'] == 'C':
            if row['consecutive_zeros'] >= zero_weeks_threshold:
                score += 3
            elif row['zero_weeks_12'] >= zero_weeks_threshold // 2:
                score += 2

            if row['no_store_ratio'] > max_store_ratio:
                score += 2

            if row['total_qty'] < min_total_sales:
                score += 2

            if row['trend'] < -0.1:
                score += 1

        # Категорія B - помірні критерії (ВИПРАВЛЕНО)
        elif row['abc_category'] == 'B':
            if row['consecutive_zeros'] >= zero_weeks_threshold * 2:  # 24 тижні
                score += 3
            elif row['consecutive_zeros'] >= zero_weeks_threshold:  # 12 тижнів
                score += 2

            if row['no_store_ratio'] > max_store_ratio:  # 85%
                score += 2

            if row['total_qty'] < min_total_sales * 2:  # 10 одиниць
                score += 1

            if row['trend'] < -0.1:
                score += 1

        # Категорія A - тільки критичні випадки
        elif row['abc_category'] == 'A':
            if row['consecutive_zeros'] >= zero_weeks_threshold * 3:  # 36 тижнів
                score += 2
            if row['no_store_ratio'] > 0.95:  # 95%
                score += 1

        # Критичні випадки для БУДЬ-ЯКОЇ категорії
        if row['consecutive_zeros'] >= zero_weeks_threshold * 2 and row['no_store_ratio'] > max_store_ratio:
            score += 2  # Посилення для комбінації факторів

        return 1 if score >= 4 else 0

    # Об'єднання даних
    final_features = features.merge(
        abc_analysis[['Art', 'total_qty', 'abc_category', 'last_sale']],
        on='Art',
        how='left'
    )
    final_features['label'] = final_features.apply(create_labels, axis=1)

    # Навчання моделі
    feature_cols = ['ma_3', 'ma_6', 'consecutive_zeros', 'zero_weeks_12', 'trend', 'no_store_ratio', 'total_qty']
    X = final_features[feature_cols].fillna(0)
    y = final_features['label']
    
    st.write(f"**Розподіл:** Зняти: {y.sum()}, Залишити: {len(y) - y.sum()}")

    # Перевірка можливості навчання (покращено: мінімум 2 зразки в кожному класі)
    if len(y.unique()) > 1 and y.sum() >= 2 and len(y) - y.sum() >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                stratify=y,
                random_state=42,
                test_size=0.3
            )

            clf = RandomForestClassifier(
                n_estimators=30,
                random_state=42,
                class_weight='balanced' if use_balanced_model else None,
                max_depth=8,
                min_samples_split=5,
                n_jobs=1
            )

            clf.fit(X_train, y_train)
            final_features['prob_dying'] = clf.predict_proba(X)[:, 1] * 100
            test_score = clf.score(X_test, y_test)

        except Exception as e:
            st.warning(f"⚠️ Помилка ML: {e}. Використовуємо просту логіку.")
            final_features['prob_dying'] = final_features['label'].astype(float) * 100
            test_score = 0.0
    else:
        st.warning("⚠️ Недостатньо даних для ML. Використовуємо просту логіку.")
        final_features['prob_dying'] = final_features['label'].astype(float) * 100
        test_score = 0.0

    return final_features, test_score

def create_prophet_forecasts(df, abc_analysis):
    if not PROPHET_AVAILABLE:
        return pd.DataFrame()
    
    try:
        with st.spinner("📈 Прогнози Prophet..."):
            top_arts = abc_analysis.nlargest(TOP_N, 'total_qty')['Art']
            forecasts = []
            
            for art in top_arts:
                try:
                    sales = df[df['Art'] == art].groupby('Data')['Qty'].sum().reset_index()
                    if len(sales) < 8: 
                        continue
                    
                    sales.columns = ['ds', 'y']
                    
                    model = Prophet(
                        daily_seasonality=False, 
                        weekly_seasonality=False, 
                        yearly_seasonality=False,
                        changepoint_prior_scale=0.05
                    )
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(sales)
                        future = model.make_future_dataframe(periods=30)
                        forecast = model.predict(future)
                    
                    median_30 = max(0, forecast.tail(30)['yhat'].median())
                    forecasts.append({'Art': art, 'forecast_30_median': float(median_30)})
                    
                except Exception as e:
                    continue
            
            return pd.DataFrame(forecasts)
            
    except Exception as e:
        st.warning(f"⚠️ Помилка Prophet: {e}")
        return pd.DataFrame()

def get_recommendations(row):
    # Формування причин
    reasons = []

    if row['abc_category'] == 'C':
        reasons.append("Категорія C")
    elif row['abc_category'] == 'B':
        reasons.append("Категорія B")

    if row['consecutive_zeros'] >= zero_weeks_threshold * 2:
        reasons.append(f"Без продажів {int(row['consecutive_zeros'])} тижнів (критично!)")
    elif row['consecutive_zeros'] >= zero_weeks_threshold:
        reasons.append(f"Без продажів {int(row['consecutive_zeros'])} тижнів")

    if row['zero_weeks_12'] >= zero_weeks_threshold // 2:
        reasons.append(f"З 12 тижнів {int(row['zero_weeks_12'])} без продажів")

    if row['no_store_ratio'] > max_store_ratio:
        stores_with_sales_pct = (1 - row['no_store_ratio']) * 100
        reasons.append(f"Продажі в {stores_with_sales_pct:.0f}% магазинів")

    if row['total_qty'] < min_total_sales:
        reasons.append(f"Малий обсяг ({row['total_qty']:.1f})")
    elif row['total_qty'] < min_total_sales * 2:
        reasons.append(f"Низький обсяг ({row['total_qty']:.1f})")

    if row['trend'] < -0.1:
        reasons.append("Негативний тренд")

    # Додаємо дату останнього продажу
    if pd.notnull(row.get('last_sale')):
        last_sale_str = row['last_sale'].strftime('%Y-%m-%d')
        reasons.append(f"Останній продаж: {last_sale_str}")

    reason = "; ".join(reasons) if reasons else "Стабільні продажі"

    # КРИТИЧНІ ВИПАДКИ - перевизначення незалежно від ML
    # 1. Екстремально тривала відсутність продажів
    if row['consecutive_zeros'] >= zero_weeks_threshold * 3:  # 36 тижнів
        return reason, "🚫 Зняти"

    # 2. Категорія C з перевищенням всіх порогів
    if (row['abc_category'] == 'C' and
        row['consecutive_zeros'] >= zero_weeks_threshold and
        row['total_qty'] < min_total_sales and
        row['no_store_ratio'] > max_store_ratio):
        return reason, "🚫 Зняти"

    # 3. Категорія B з критичними показниками
    if (row['abc_category'] == 'B' and
        row['consecutive_zeros'] >= zero_weeks_threshold * 2 and
        row['no_store_ratio'] > max_store_ratio):
        return reason, "🚫 Зняти"

    # 4. Тривала відсутність + низьке поширення для B
    if (row['abc_category'] == 'B' and
        row['consecutive_zeros'] >= zero_weeks_threshold * 1.5 and
        row['no_store_ratio'] > 0.85 and
        row['total_qty'] < min_total_sales * 2):
        return reason, "⚠️ Спостерігати"

    # Стандартна логіка на основі ML
    prob_threshold_pct = final_threshold * 100

    if row['prob_dying'] > prob_threshold_pct:
        return reason, "🚫 Зняти"
    elif row['prob_dying'] > prob_threshold_pct * 0.7:
        return reason, "⚠️ Спостерігати"

    # Додаткові перевірки для "Спостерігати"
    if (row['consecutive_zeros'] >= zero_weeks_threshold and
        row['no_store_ratio'] > 0.75):
        return reason, "⚠️ Спостерігати"

    return reason, "✅ Залишити"

# Виконання аналізу
df, weekly, all_arts, unique_weeks = process_data(df)
abc_analysis = calculate_abc_xyz_analysis(df)
features = calculate_features(weekly, df)
final_features, test_score = create_ml_model(features, abc_analysis)
forecast_df = create_prophet_forecasts(df, abc_analysis)

# Фінальна таблиця
final = final_features.merge(abc_analysis[['Art', 'xyz_category', 'last_sale']], on='Art', how='left')

# Перевірка перед мерджем forecast_df
if not forecast_df.empty:
    final = final.merge(forecast_df, on='Art', how='left')

# Обробка пустих Name
final = final.merge(df[['Art', 'Name']].drop_duplicates(), on='Art', how='left')
final['Name'] = final['Name'].fillna('Без назви')

# Отримання рекомендацій
recommendations = final.apply(get_recommendations, axis=1)
final['Причина'] = [rec[0] for rec in recommendations]
final['Рекомендація'] = [rec[1] for rec in recommendations]

# === РЕЗУЛЬТАТИ ===
st.header("📊 Результати аналізу")

total_products = len(final)
candidates_remove = len(final[final['Рекомендація'] == "🚫 Зняти"])
candidates_watch = len(final[final['Рекомендація'] == "⚠️ Спостерігати"])
candidates_keep = len(final[final['Рекомендація'] == "✅ Залишити"])

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Всього товарів", total_products)
with col2: st.metric("До зняття", candidates_remove, f"{candidates_remove/total_products*100:.1f}%")
with col3: st.metric("Спостерігати", candidates_watch, f"{candidates_watch/total_products*100:.1f}%")
with col4: st.metric("Точність моделі", f"{test_score:.2f}" if test_score > 0 else "N/A")

# ABC/XYZ розподіл
st.subheader("📈 ABC/XYZ аналіз")
abc_dist = final['abc_category'].value_counts()
xyz_dist = final['xyz_category'].value_counts()

col1, col2 = st.columns(2)
with col1:
    st.write("**ABC категорії:**")
    st.write(f"A: {abc_dist.get('A', 0)}, B: {abc_dist.get('B', 0)}, C: {abc_dist.get('C', 0)}")
with col2:
    st.write("**XYZ категорії:**")
    st.write(f"X: {xyz_dist.get('X', 0)}, Y: {xyz_dist.get('Y', 0)}, Z: {xyz_dist.get('Z', 0)}")

# === НОВИЙ РОЗДІЛ: СТАТИСТИКА ДЛЯ ПРОДАЖІВ ТА МАРКЕТИНГУ ===
st.header("📈 Аналітика для відділу продажів та маркетингу")

# Розрахунок додаткових метрик
total_sales_volume = final['total_qty'].sum()
remove_sales_volume = final[final['Рекомендація'] == "🚫 Зняти"]['total_qty'].sum()
watch_sales_volume = final[final['Рекомендація'] == "⚠️ Спостерігати"]['total_qty'].sum()
keep_sales_volume = final[final['Рекомендація'] == "✅ Залишити"]['total_qty'].sum()

# 1. Зведена таблиця за рекомендаціями та ABC
st.subheader("📊 Зведена таблиця: Рекомендації × ABC категорії")

summary_pivot = pd.crosstab(
    final['Рекомендація'],
    final['abc_category'],
    values=final['total_qty'],
    aggfunc='sum',
    margins=True,
    margins_name='Разом'
).fillna(0).astype(int)

st.dataframe(summary_pivot.style.format("{:,}"), use_container_width=True)

# 2. Таблиця з ключовими метриками
st.subheader("💼 Ключові бізнес-метрики")

metrics_data = {
    'Категорія': ['🚫 Зняти', '⚠️ Спостерігати', '✅ Залишити', '**РАЗОМ**'],
    'Кількість товарів': [candidates_remove, candidates_watch, candidates_keep, total_products],
    '% від асортименту': [
        f"{candidates_remove/total_products*100:.1f}%",
        f"{candidates_watch/total_products*100:.1f}%",
        f"{candidates_keep/total_products*100:.1f}%",
        "100%"
    ],
    'Обсяг продажів (од.)': [
        f"{remove_sales_volume:,.0f}",
        f"{watch_sales_volume:,.0f}",
        f"{keep_sales_volume:,.0f}",
        f"{total_sales_volume:,.0f}"
    ],
    '% від обороту': [
        f"{remove_sales_volume/total_sales_volume*100:.1f}%",
        f"{watch_sales_volume/total_sales_volume*100:.1f}%",
        f"{keep_sales_volume/total_sales_volume*100:.1f}%",
        "100%"
    ]
}

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# 3. Топ-20 товарів до зняття
st.subheader("🔴 Топ-20 товарів до зняття (за обсягом продажів)")

remove_candidates = final[final['Рекомендація'] == "🚫 Зняти"].nlargest(20, 'total_qty')
remove_display = remove_candidates[['Art', 'Name', 'abc_category', 'total_qty', 'consecutive_zeros', 'no_store_ratio', 'Причина']].copy()
remove_display['no_store_ratio'] = (remove_display['no_store_ratio'] * 100).round(1).astype(str) + '%'
remove_display.columns = ['Артикул', 'Назва', 'ABC', 'Обсяг продажів', 'Тижнів без продажів', 'Магазинів без продажів', 'Причина']

st.dataframe(remove_display, use_container_width=True, hide_index=True)

# 4. Товари під спостереженням
st.subheader("🟡 Топ-20 товарів під спостереженням")

watch_candidates = final[final['Рекомендація'] == "⚠️ Спостерігати"].nlargest(20, 'total_qty')
watch_display = watch_candidates[['Art', 'Name', 'abc_category', 'total_qty', 'consecutive_zeros', 'prob_dying', 'Причина']].copy()
watch_display['prob_dying'] = watch_display['prob_dying'].round(1).astype(str) + '%'
watch_display.columns = ['Артикул', 'Назва', 'ABC', 'Обсяг продажів', 'Тижнів без продажів', 'Ризик зняття', 'Причина']

st.dataframe(watch_display, use_container_width=True, hide_index=True)

# 5. Статистика по магазинах
st.subheader("🏪 Розподіл продажів по магазинах")

store_stats = df.groupby('Magazin').agg({
    'Art': 'nunique',
    'Qty': 'sum'
}).reset_index()
store_stats.columns = ['Магазин', 'Унікальних товарів', 'Обсяг продажів']
store_stats = store_stats.sort_values('Обсяг продажів', ascending=False)

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(store_stats, use_container_width=True, hide_index=True)
with col2:
    st.metric("Всього магазинів", len(store_stats))
    st.metric("Середній оборот", f"{store_stats['Обсяг продажів'].mean():,.0f} од.")

# === ФІЛЬТРИ І ТАБЛИЦЯ ===
st.subheader("🔍 Фільтри")
col1, col2, col3 = st.columns(3)

with col1:
    filter_recommendation = st.selectbox("Рекомендація:", ["Всі", "🚫 Зняти", "⚠️ Спостерігати", "✅ Залишити"])
    filter_abc = st.selectbox("ABC:", ["Всі", "A", "B", "C"])
with col2:
    min_prob = st.slider("Мін. ймовірність (%)", 0, 100, 0)
    filter_xyz = st.selectbox("XYZ:", ["Всі", "X", "Y", "Z"])
with col3:
    min_zero_weeks = st.slider("Мін. тижнів без продажів", 0, 20, 0)
    search_art = st.text_input("Пошук артикула/назви")

# Застосування фільтрів
filtered_df = final.copy()
if filter_recommendation != "Всі":
    filtered_df = filtered_df[filtered_df['Рекомендація'] == filter_recommendation]
if filter_abc != "Всі":
    filtered_df = filtered_df[filtered_df['abc_category'] == filter_abc]
if filter_xyz != "Всі":
    filtered_df = filtered_df[filtered_df['xyz_category'] == filter_xyz]

filtered_df = filtered_df[
    (filtered_df['prob_dying'] >= min_prob) &
    (filtered_df['consecutive_zeros'] >= min_zero_weeks)
]

if search_art:
    mask = (filtered_df['Art'].astype(str).str.contains(search_art, case=False, na=False) |
            filtered_df['Name'].astype(str).str.contains(search_art, case=False, na=False))
    filtered_df = filtered_df[mask]

# Таблиця результатів
st.subheader(f"📋 Результати ({len(filtered_df)} товарів)")

display_columns = ['Art', 'Name', 'abc_category', 'xyz_category', 'total_qty', 'consecutive_zeros', 'no_store_ratio', 'prob_dying', 'Причина', 'Рекомендація']
if 'forecast_30_median' in filtered_df.columns:
    display_columns.insert(-2, 'forecast_30_median')

display_df = filtered_df[display_columns].copy()
display_df['no_store_ratio'] = (display_df['no_store_ratio'] * 100).round(1)
display_df['prob_dying'] = display_df['prob_dying'].round(1)

column_names = ['Артикул', 'Назва', 'ABC', 'XYZ', 'Обсяг', 'Тижнів_без_продажів', 'Магазини_без_продажів_%', 'Ймовірність_зняття_%']
if 'forecast_30_median' in display_df.columns:
    column_names.append('Прогноз_30дн')
column_names.extend(['Причина', 'Рекомендація'])

display_df.columns = column_names
st.dataframe(display_df, use_container_width=True)

# === ЕКСПОРТ ===
st.subheader("💾 Експорт")
if st.button("📥 Підготувати Excel"):
    try:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            output_cols = ['Art', 'Name', 'abc_category', 'xyz_category', 'total_qty', 'consecutive_zeros', 'no_store_ratio', 'prob_dying', 'Причина', 'Рекомендація']
            if 'forecast_30_median' in final.columns:
                output_cols.insert(-2, 'forecast_30_median')

            final[output_cols].to_excel(writer, sheet_name='Результати', index=False)

            stats = pd.DataFrame({
                'Метрика': ['Всього', 'Зняти', 'Спостерігати', 'Залишити', 'Поріг_ML_%'],
                'Значення': [total_products, candidates_remove, candidates_watch,
                           total_products - candidates_remove - candidates_watch, final_threshold*100]
            })
            stats.to_excel(writer, sheet_name='Статистика', index=False)

            # Зведена таблиця
            summary_pivot.to_excel(writer, sheet_name='Зведена_ABC')

            # Бізнес-метрики
            metrics_df.to_excel(writer, sheet_name='Бізнес_метрики', index=False)

            # Топ до зняття
            if len(remove_display) > 0:
                remove_display.to_excel(writer, sheet_name='Топ_до_зняття', index=False)

        st.download_button("📥 Завантажити Excel", buffer.getvalue(), "analysis_results.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success("✅ Готово!")
    except Exception as e:
        st.error(f"❌ Помилка: {str(e)}")

with st.expander("ℹ️ Інформація"):
    st.write(f"**Статус:** Prophet {'✅' if PROPHET_AVAILABLE else '❌'}, Оброблено: {len(final)}")
    if not PROPHET_AVAILABLE:
        st.warning("⚠️ Встановіть Prophet: pip install prophet")

st.divider()
st.caption("📊 Звіт згенеровано системою аналізу товарного портфеля")
