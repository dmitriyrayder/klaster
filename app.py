import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Кластеризация магазинов", layout="wide")

st.title("📊 Кластеризация магазинов по структуре ассортимента")
st.markdown("**Метод:** Сегментация по долям товарных сегментов в обороте")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл с продажами (Excel)", type=['xlsx', 'xls'])

if uploaded_file:
    # Чтение данных
    df = pd.read_excel(uploaded_file)
    
    st.success(f"✅ Загружено: {len(df):,} строк, {df['Magazin'].nunique()} магазинов, {df['Art'].nunique():,} артикулов")
    
    # Проверка колонок
    required_cols = ['Magazin', 'Segment', 'Sum']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Файл должен содержать колонки: {required_cols}")
        st.stop()
    
    # --- БЛОК 1: АНАЛИЗ СЕГМЕНТОВ ---
    st.header("1️⃣ Анализ товарных сегментов")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Структура оборота по сегментам")
        segment_sales = df.groupby('Segment')['Sum'].sum().sort_values(ascending=False)
        segment_pct = (segment_sales / segment_sales.sum() * 100).round(2)
        
        segment_df = pd.DataFrame({
            'Сегмент': segment_sales.index,
            'Оборот, ₴': segment_sales.values,
            'Доля, %': segment_pct.values
        })
        st.dataframe(segment_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Распределение оборота")
        fig_pie = px.pie(segment_df, values='Оборот, ₴', names='Сегмент', 
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # --- БЛОК 2: ПОСТРОЕНИЕ МАТРИЦЫ ---
    st.header("2️⃣ Матрица магазин × сегмент")
    
    # Агрегируем продажи по магазинам и сегментам
    pivot = df.groupby(['Magazin', 'Segment'])['Sum'].sum().reset_index()
    pivot_table = pivot.pivot(index='Magazin', columns='Segment', values='Sum').fillna(0)
    
    # Вычисляем доли сегментов для каждого магазина
    pivot_pct = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100
    
    # Проверка на достаточное количество магазинов
    n_stores = len(pivot_pct)
    if n_stores < 3:
        st.error(f"❌ Недостаточно магазинов для кластеризации: {n_stores}. Минимум: 3")
        st.stop()
    
    st.subheader("Доля сегментов в обороте каждого магазина (%)")
    st.dataframe(pivot_pct.round(2).style.background_gradient(cmap='RdYlGn', axis=None), 
                 use_container_width=True)
    
    # Стандартизация данных (используется во всех последующих блоках)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot_pct)
    
    # --- БЛОК 3: ПОДБОР ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ ---
    st.header("3️⃣ Подбор оптимального количества кластеров")
    
    with st.expander("⚙️ Настройки анализа", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            min_k = st.number_input("Min кластеров", min_value=2, max_value=min(10, n_stores-1), value=2)
        with col_s2:
            max_k = st.number_input("Max кластеров", min_value=2, max_value=min(15, n_stores-1), value=min(10, n_stores-1))
        with col_s3:
            init_method = st.selectbox("Метод инициализации", ['k-means++', 'random'], index=0)
    
    # Вычисляем метрики для разного количества кластеров
    k_range = range(min_k, max_k + 1)
    
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    inertias = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, k in enumerate(k_range):
        status_text.text(f"Анализ {k} кластеров...")
        kmeans_temp = KMeans(n_clusters=k, random_state=42, init=init_method, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        
        silhouette_scores.append(silhouette_score(X_scaled, labels_temp))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels_temp))
        calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels_temp))
        inertias.append(kmeans_temp.inertia_)
        
        progress_bar.progress((i + 1) / len(k_range))
    
    progress_bar.empty()
    status_text.empty()
    
    # Оптимальное количество кластеров
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_davies = k_range[np.argmin(davies_bouldin_scores)]
    optimal_k_calinski = k_range[np.argmax(calinski_harabasz_scores)]
    
    # Elbow method - находим точку максимального изгиба (максимальное абсолютное изменение)
    if len(inertias) > 2:
        inertia_diffs = np.abs(np.diff(inertias))
        optimal_k_elbow_idx = np.argmax(inertia_diffs)
        optimal_k_elbow = list(k_range)[optimal_k_elbow_idx]
    else:
        optimal_k_elbow = min_k + 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Silhouette", f"k={optimal_k_silhouette}", 
                  help="Максимальная разделимость кластеров")
    with col2:
        st.metric("🏆 Davies-Bouldin", f"k={optimal_k_davies}", 
                  help="Минимальное внутрикластерное расстояние")
    with col3:
        st.metric("🏆 Calinski-Harabasz", f"k={optimal_k_calinski}", 
                  help="Максимальная дисперсия между кластерами")
    with col4:
        st.metric("🏆 Elbow Method", f"k={optimal_k_elbow}", 
                  help="Точка максимального изгиба")
    
    # Графики метрик
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Silhouette & Davies-Bouldin
        fig_metrics1 = go.Figure()
        fig_metrics1.add_trace(go.Scatter(
            x=list(k_range), y=silhouette_scores, mode='lines+markers',
            name='Silhouette (↑ лучше)', line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        fig_metrics1.add_trace(go.Scatter(
            x=list(k_range), y=davies_bouldin_scores, mode='lines+markers',
            name='Davies-Bouldin (↓ лучше)', line=dict(color='red', width=3),
            marker=dict(size=8), yaxis='y2'
        ))
        fig_metrics1.update_layout(
            title="Метрики качества кластеризации",
            xaxis_title="Количество кластеров",
            yaxis_title="Silhouette Score",
            yaxis2=dict(title="Davies-Bouldin Index", overlaying='y', side='right'),
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_metrics1, use_container_width=True)
    
    with col_g2:
        # Elbow method
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=inertias, mode='lines+markers',
            name='Inertia', line=dict(color='blue', width=3),
            marker=dict(size=10, color=inertias, colorscale='Viridis', showscale=True)
        ))
        fig_elbow.add_vline(x=optimal_k_elbow, line_dash="dash", line_color="red",
                           annotation_text=f"Оптимум: k={optimal_k_elbow}")
        fig_elbow.update_layout(
            title="Elbow Method (метод локтя)",
            xaxis_title="Количество кластеров",
            yaxis_title="Inertia (сумма квадратов расстояний)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    # Таблица всех метрик
    with st.expander("📊 Детальная таблица метрик"):
        metrics_df = pd.DataFrame({
            'K': list(k_range),
            'Silhouette': [f"{x:.4f}" for x in silhouette_scores],
            'Davies-Bouldin': [f"{x:.4f}" for x in davies_bouldin_scores],
            'Calinski-Harabasz': [f"{x:.0f}" for x in calinski_harabasz_scores],
            'Inertia': [f"{x:.2f}" for x in inertias]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # ИСПРАВЛЕНО: правильная индексация
    silhouette_optimal_idx = optimal_k_silhouette - min_k
    st.info(f"""
    **Рекомендация:** Оптимальное количество кластеров — **{optimal_k_silhouette}** 
    (по Silhouette Score: {silhouette_scores[silhouette_optimal_idx]:.3f})
    """)
    
    # --- БЛОК 4: КЛАСТЕРИЗАЦИЯ ---
    st.header("4️⃣ Кластеризация магазинов")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        n_clusters = st.slider("Количество кластеров", min_value=2, max_value=min(10, n_stores-1), value=optimal_k_silhouette)
    
    with col2:
        random_state = st.number_input("Random state", value=42, min_value=0)
    
    with col3:
        max_iter = st.number_input("Max iterations", value=300, min_value=100, max_value=1000, step=100)
    
    with col4:
        distance_metric = st.selectbox("Расстояние", ['euclidean', 'manhattan'], 
                                       help="Метрика расстояния между точками")
    
    # Кластеризация
    # ИСПРАВЛЕНО: раздельная обработка для разных алгоритмов
    if distance_metric == 'euclidean':
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, 
                       init=init_method, n_init=10, max_iter=max_iter)
        clusters = kmeans.fit_predict(X_scaled)
        has_inertia = True
    else:
        # Для Manhattan используем иерархическую кластеризацию
        from sklearn.cluster import AgglomerativeClustering
        kmeans = AgglomerativeClustering(n_clusters=n_clusters, metric='manhattan', linkage='average')
        clusters = kmeans.fit_predict(X_scaled)
        has_inertia = False
    
    # Метрики качества
    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Silhouette Score", f"{silhouette:.3f}", 
                  help="0.5-0.7: хорошо, >0.7: отлично")
    with col_m2:
        st.metric("Davies-Bouldin", f"{davies_bouldin:.3f}",
                  help="Чем меньше, тем лучше. <1.0: отлично")
    with col_m3:
        st.metric("Calinski-Harabasz", f"{calinski_harabasz:.0f}",
                  help="Чем больше, тем лучше")
    with col_m4:
        # ИСПРАВЛЕНО: корректная проверка наличия inertia
        if has_inertia:
            st.metric("Inertia", f"{kmeans.inertia_:.2f}",
                      help="Сумма квадратов расстояний")
        else:
            st.metric("Метод", "Agglomerative", help="Иерархическая кластеризация")
    
    # Добавляем кластеры в данные
    # ИСПРАВЛЕНО: создаем копию для избежания проблем с индексацией
    pivot_pct_clustered = pivot_pct.copy()
    pivot_pct_clustered['Кластер'] = clusters
    pivot_pct_clustered = pivot_pct_clustered.sort_values('Кластер')
    
    # --- БЛОК 5: ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ В 2D (PCA) ---
    st.subheader("Визуализация кластеров в 2D (PCA)")
    
    col_v1, col_v2 = st.columns([2, 1])
    
    with col_v1:
        # PCA для визуализации
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Кластер': [f"Кластер {c}" for c in clusters],
            'Магазин': pivot_pct.index
        })
        
        fig_pca = px.scatter(
            pca_df, x='PC1', y='PC2', color='Кластер',
            hover_data=['Магазин'],
            title=f"Кластеры в пространстве главных компонент (объясненная дисперсия: {pca.explained_variance_ratio_.sum():.1%})",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pca.update_traces(marker=dict(size=12, line=dict(width=2, color='white')))
        fig_pca.update_layout(height=500)
        st.plotly_chart(fig_pca, use_container_width=True)
    
    with col_v2:
        st.markdown("**Объясненная дисперсия:**")
        variance_df = pd.DataFrame({
            'Компонента': ['PC1', 'PC2'],
            'Дисперсия, %': [f"{x*100:.1f}%" for x in pca.explained_variance_ratio_]
        })
        st.dataframe(variance_df, use_container_width=True, hide_index=True)
        
        st.markdown("**Интерпретация:**")
        st.markdown(f"""
        - PC1: {pca.explained_variance_ratio_[0]*100:.1f}% вариации
        - PC2: {pca.explained_variance_ratio_[1]*100:.1f}% вариации
        - Близкие точки = похожие магазины
        """)
    
    # --- БЛОК 6: ПРОФИЛИ КЛАСТЕРОВ ---
    st.subheader("Профили кластеров")
    
    # ИСПРАВЛЕНО: используем копию без колонки Оборот
    cluster_profiles = pivot_pct_clustered.drop(columns=['Кластер'], errors='ignore').groupby(pivot_pct_clustered['Кластер']).mean()
    
    # Тепловая карта
    fig_heatmap = px.imshow(
        cluster_profiles.T, 
        labels=dict(x="Кластер", y="Сегмент", color="Доля, %"),
        x=[f"Кластер {i}" for i in range(n_clusters)],
        y=cluster_profiles.columns,
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # --- БЛОК 7: СТАТИСТИКА ПО КЛАСТЕРАМ ---
    st.header("7️⃣ Характеристика кластеров")
    
    # Добавляем оборот магазинов
    store_totals = df.groupby('Magazin')['Sum'].sum()
    pivot_pct_clustered['Оборот_магазина'] = pivot_pct_clustered.index.map(store_totals)
    
    for cluster_id in range(n_clusters):
        with st.expander(f"**Кластер {cluster_id}** ({(clusters == cluster_id).sum()} магазинов)", expanded=True):
            cluster_data = pivot_pct_clustered[pivot_pct_clustered['Кластер'] == cluster_id]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Магазины в кластере:**")
                stores_list = cluster_data.index.tolist()
                st.write(", ".join(stores_list))
            
            with col2:
                total_revenue = cluster_data['Оборот_магазина'].sum()
                st.metric("Суммарный оборот", f"{total_revenue:,.0f} ₴")
            
            st.markdown("**Средний профиль кластера (доля сегментов, %):**")
            
            # Средние доли сегментов
            profile = cluster_data.drop(['Кластер', 'Оборот_магазина'], axis=1).mean().sort_values(ascending=False)
            
            profile_df = pd.DataFrame({
                'Сегмент': profile.index,
                'Средняя доля, %': profile.values.round(2)
            })
            
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.dataframe(profile_df, use_container_width=True, hide_index=True)
            
            with col_b:
                fig_bar = px.bar(profile_df, x='Средняя доля, %', y='Сегмент', 
                                orientation='h', color='Средняя доля, %',
                                color_continuous_scale='Viridis')
                fig_bar.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
    
    # --- БЛОК 8: ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ (ДЕНДРОГРАММА) ---
    st.header("8️⃣ Дендрограмма (иерархическая кластеризация)")
    
    with st.expander("📊 Показать дендрограмму", expanded=False):
        linkage_method = st.selectbox("Метод связи", ['ward', 'average', 'complete', 'single'])
        
        # Вычисляем linkage matrix
        Z = linkage(X_scaled, method=linkage_method)
        
        # Создаем дендрограмму
        fig_dendr = go.Figure()
        
        dendr = dendrogram(Z, labels=pivot_pct.index.tolist(), no_plot=True)
        
        icoord = np.array(dendr['icoord'])
        dcoord = np.array(dendr['dcoord'])
        
        for i in range(len(icoord)):
            fig_dendr.add_trace(go.Scatter(
                x=icoord[i], y=dcoord[i],
                mode='lines',
                line=dict(color='rgb(100,100,100)', width=1),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # ИСПРАВЛЕНО: правильные позиции для меток
        leaves_positions = dendr['leaves']
        leaves_labels = [pivot_pct.index[i] for i in leaves_positions]
        tick_positions = [5 + i*10 for i in range(len(leaves_labels))]
        
        fig_dendr.update_layout(
            title="Дендрограмма: иерархия схожести магазинов",
            xaxis=dict(title="Магазины", ticktext=leaves_labels, 
                      tickvals=tick_positions),
            yaxis_title="Расстояние",
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_dendr, use_container_width=True)
        
        st.info("""
        **Как читать:** Чем ниже точка слияния, тем более похожи магазины.
        Вертикальная линия = группа схожих магазинов.
        """)
    
    # --- БЛОК 9: СРАВНЕНИЕ МАГАЗИНОВ ---
    st.header("9️⃣ Поиск похожих магазинов")
    
    col_c1, col_c2 = st.columns([1, 2])
    
    with col_c1:
        selected_store = st.selectbox("Выберите магазин:", pivot_pct.index.tolist())
    
    if selected_store:
        store_cluster = pivot_pct_clustered.loc[selected_store, 'Кластер']
        
        # Извлекаем профиль магазина
        store_profile = pivot_pct.loc[selected_store]
        
        # Находим самые похожие магазины (по косинусному расстоянию)
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Вычисляем схожесть со всеми магазинами
        similarities = cosine_similarity([store_profile], pivot_pct)[0]
        
        # Создаем DataFrame для удобной работы
        similarity_df = pd.DataFrame({
            'Магазин': pivot_pct.index,
            'Схожесть': similarities,
            'Кластер': pivot_pct_clustered['Кластер'].values
        })
        
        # КРИТИЧНО: Явно исключаем выбранный магазин
        similarity_df = similarity_df[similarity_df['Магазин'] != selected_store]
        
        # Сортируем по схожести и берем топ-5
        similarity_df = similarity_df.sort_values('Схожесть', ascending=False).head(5)
        
        similar_stores = similarity_df['Магазин'].values
        similar_scores = similarity_df['Схожесть'].values
        
        with col_c2:
            st.markdown(f"**Кластер:** {int(store_cluster)}")
            st.markdown("**Топ-5 похожих магазинов:**")
            
            # Форматируем для отображения
            display_df = similarity_df[['Магазин', 'Схожесть', 'Кластер']].copy()
            display_df['Схожесть'] = display_df['Схожесть'].apply(lambda x: f"{x*100:.1f}%")
            display_df['Кластер'] = display_df['Кластер'].astype(int)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Сравнение профилей
        st.markdown("**Сравнение профилей (доли сегментов):**")
        
        comparison_data = []
        comparison_data.append(store_profile.values)
        for store in similar_stores[:3]:
            comparison_data.append(pivot_pct.loc[store].values)
        
        comparison_df = pd.DataFrame(
            comparison_data,
            columns=store_profile.index,
            index=[selected_store] + list(similar_stores[:3])
        ).T
        
        fig_compare = px.bar(
            comparison_df,
            barmode='group',
            title="Сравнение структуры ассортимента",
            labels={'value': 'Доля, %', 'index': 'Сегмент'}
        )
        fig_compare.update_layout(height=400)
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # --- БЛОК 10: РЕКОМЕНДАЦИИ ---
    st.header("🎯 Рекомендации по оптимизации")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("### По результатам кластеризации:")
        st.markdown(f"""
        1. **Создайте {n_clusters} торговые матрицы** — по одной на кластер
        2. **Флагманские магазины** — кластеры с высокой долей премиум-сегмента
        3. **Формат "у дома"** — кластеры с фокусом на эконом-сегмент
        4. **Тестирование** — перенос ассортимента между похожими магазинами
        5. **Мониторинг** — повторная кластеризация каждые 3-6 месяцев
        """)
    
    with rec_col2:
        st.markdown("### Качество модели:")
        
        quality_status = "🟢 Отличное" if silhouette > 0.7 else "🟡 Хорошее" if silhouette > 0.5 else "🔴 Требует улучшения"
        st.markdown(f"**Статус:** {quality_status}")
        
        if silhouette < 0.5:
            st.warning("""
            **Рекомендации по улучшению:**
            - Попробуйте изменить количество кластеров
            - Используйте метод иерархической кластеризации
            - Добавьте дополнительные признаки (оборот, ABC-категории)
            """)
        
        st.markdown(f"""
        **Метрики:**
        - Silhouette: {silhouette:.3f} {'✓' if silhouette > 0.5 else '✗'}
        - Davies-Bouldin: {davies_bouldin:.3f} {'✓' if davies_bouldin < 1.0 else '✗'}
        - Calinski-Harabasz: {calinski_harabasz:.0f}
        """)
    
    # --- БЛОК 11: EXPORT ---
    st.header("📥 Экспорт результатов")
    
    # Подготовка итоговой таблицы
    result_df = pivot_pct_clustered.reset_index()
    result_df = result_df.rename(columns={'index': 'Магазин'})
    
    # Добавляем метрики качества в экспорт
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # CSV экспорт
        csv = result_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Скачать результаты кластеризации (CSV)",
            data=csv,
            file_name=f"store_clusters_k{n_clusters}.csv",
            mime="text/csv"
        )
    
    with export_col2:
        # Excel экспорт с несколькими листами
        from io import BytesIO
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name='Кластеры', index=False)
            
            # Профили кластеров
            cluster_profiles.to_excel(writer, sheet_name='Профили_кластеров')
            
            # Метрики
            metrics_summary = pd.DataFrame({
                'Метрика': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score'],
                'Значение': [silhouette, davies_bouldin, calinski_harabasz],
                'Интерпретация': [
                    '>0.5: хорошо, >0.7: отлично',
                    '<1.0: отлично',
                    'Чем больше, тем лучше'
                ]
            })
            metrics_summary.to_excel(writer, sheet_name='Метрики', index=False)
        
        st.download_button(
            label="📥 Скачать полный отчет (Excel)",
            data=output.getvalue(),
            file_name=f"store_clustering_report_k{n_clusters}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    st.markdown("---")
    st.success(f"""
    ✅ **Анализ завершен!** 
    
    - Проанализировано: {len(pivot_pct)} магазинов
    - Создано: {n_clusters} кластеров
    - Качество (Silhouette): {silhouette:.3f} {'🟢' if silhouette > 0.7 else '🟡' if silhouette > 0.5 else '🔴'}
    - Рекомендуемые кластеры: {optimal_k_silhouette} (по всем метрикам)
    """)
    
    st.markdown("---")
    
    with st.expander("ℹ️ Справка по интерпретации результатов"):
        st.markdown("""
        ### Метрики качества кластеризации
        
        **Silhouette Score** (коэффициент силуэта)
        - Диапазон: [-1, 1]
        - > 0.7: отличная кластеризация
        - 0.5-0.7: хорошая кластеризация
        - 0.25-0.5: приемлемая, есть наложения
        - < 0.25: плохая кластеризация
        
        **Davies-Bouldin Index**
        - Диапазон: [0, ∞)
        - < 1.0: отличная кластеризация
        - 1.0-2.0: хорошая кластеризация
        - > 2.0: слабая кластеризация
        
        **Calinski-Harabasz Score**
        - Диапазон: [0, ∞)
        - Чем больше, тем лучше
        - Нет абсолютных порогов, сравнивайте разные k
        
        **Elbow Method**
        - Ищет "локоть" на графике Inertia
        - Точка, где добавление кластеров не дает улучшения
        
        ### Применение результатов
        
        1. **Создание торговых матриц:** для каждого кластера своя матрица
        2. **Оптимизация закупок:** общие закупки для кластера
        3. **A/B тестирование:** внутри кластера магазины взаимозаменяемы
        4. **Прогнозирование:** модели на уровне кластера точнее
        5. **Управление персоналом:** обучение с учетом специфики кластера
        """)
    
    # Дополнительная информация
    st.info("""
    **💡 Советы:**
    - Запускайте анализ каждые 3-6 месяцев
    - Сравнивайте результаты при разных k через метрики
    - Используйте дендрограмму для понимания иерархии
    - Проверяйте похожие магазины для cross-selling идей
    """)

else:
    st.info("👆 Загрузите файл Excel с продажами для начала анализа")
    
    with st.expander("ℹ️ Требования к файлу"):
        st.markdown("""
        Файл должен содержать колонки:
        - **Magazin** — название магазина
        - **Segment** — товарный сегмент
        - **Sum** — сумма продаж
        
        Опционально: `Art` (артикул), `Qty` (количество)
        """)
