#################################################
#                                               #
#  IA003 - Probabilidade e Estatística para IA  #
#                                               #
#  Atividade Avaliativa                         #
#                                               #
#################################################


import os
import altair as alt
#import folium
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
#import geopandas as gpd

from datetime import date, time, timedelta
from folium.plugins import MarkerCluster, HeatMap
from html import escape
#from IPython.display import display, FileLink
from streamlit_folium import st_folium
#from shapely.geometry import Point


st.set_page_config(
    page_title="IA003 - Acidentes PoA",
    page_icon="📊",
    layout="wide",
)

# Título do dashboard

header_box = st.container()

with header_box:
    
    st.markdown("# IA003 - Probabilidade e Estatística para IA")
    st.markdown("## Dataset Acidentes de Trânsito em Porto Alegre")

    # Seletor do tipo de visualização

    pagina = st.pills(
        "Navegação",
        options=["🌐 População", "🧪 Amostra", "🔮 Inferência 1", "🔮 Inferência 2", "🔮 Inferência 3", "🔮 Inferência 4", "📈 Gráficos"],
        default="🌐 População", 
        selection_mode="single",
        label_visibility="collapsed"
    )

    # Normaliza a seleção para remover o emoji ao usar na lógica do app
    pagina_selecionada = pagina.split(" ", 1)[-1] if pagina else "Gráficos"

    # Invisible anchor div to target with CSS
    st.markdown("<div class='sticky-top-anchor'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        /* Remova o padding superior da área principal do Streamlit para o título encostar na borda */
        .main .block-container {
            padding-top: 0rem !important;
        }

        /* Fixa o container do cabeçalho diretamente na borda superior da janela */
        div[data-testid="stVerticalBlock"] div:has(div.sticky-top-anchor) {
            position: sticky;
            top: 0 !important; /* Cola no topo da janela */
            background-color: var(--background-color, white);
            z-index: 999;
            padding: 0.2rem 1rem 0.2rem 1rem; /* topo, direita, baixo, esquerda */
        }

        /* Zera as margens e paddings superiores do título */
        div[data-testid="stVerticalBlock"] div:has(div.sticky-top-anchor) h1 {
            margin-top: 0rem !important;
            padding-top: 0rem !important;
        }

        /* Zera as margens inferiores do componente de pílulas */
        div[data-testid="stPills"], div[data-testid="stElementContainer"]:has(div[data-testid="stPills"]) {
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Carga e preparação dos dados

@st.cache_data
def carregar_dados():

    # URL direta de conteúdo bruto (raw) do GitHub
    file_path = "https://raw.githubusercontent.com/dirceugrundler-maker/dashboard_statistics/main/data/cat_acidentes_2026-09-22.csv"

    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo do GitHub: {e}")
        st.stop()

    # Converte a coluna 'data' para tipo datetime
    df['data'] = pd.to_datetime(df['data'])

    # Converte a coluna 'hora' para tipo datetime
    df["hora"] = pd.to_datetime(df["hora"], format="%H:%M:%S")

    # Filtra o dataframe para incluir somente dados até 30/junho/2026
    df_2026_S1 = df[df['data'] <= '2026-06-30']

    # Monta dinamicamente a lista de tipos de veículos disponíveis no dataset
    tipos_veiculos = {}
    
    if (df_2026_S1["auto"] > 0).any():
        tipos_veiculos["Auto"] = ["auto"]
    
    if (df_2026_S1["bicicleta"] > 0).any():
        tipos_veiculos["Bicicleta"] = ["bicicleta"]
    
    if (df_2026_S1["caminhao"] > 0).any():
        tipos_veiculos["Caminhão"] = ["caminhao"]
    
    if (df_2026_S1["carroca"] > 0).any():
        tipos_veiculos["Carroça"] = ["carroca"]
    
    if (df_2026_S1["lotacao"] > 0).any():
        tipos_veiculos["Lotação"] = ["lotacao"]
    
    if (df_2026_S1["moto"] > 0).any():
        tipos_veiculos["Moto"] = ["moto"]

    if ((df_2026_S1["onibus_urb"] > 0).any() or
        (df_2026_S1["onibus_met"] > 0).any() or
        (df_2026_S1["onibus_int"] > 0).any()):
        tipos_veiculos["Ônibus"] = ["onibus_urb", "onibus_met", "onibus_int"]
    
    if (df_2026_S1["patinete"] > 0).any():
        tipos_veiculos["Patinete"] = ["patinete"]
    
    if (df_2026_S1["taxi"] > 0).any():
        tipos_veiculos["Taxi"] = ["taxi"]
    
    if (df_2026_S1["outro"] > 0).any():
        tipos_veiculos["Outro"] = ["outro"]

    # Estrutura com os tipos de acidentes e cores fortes/contrastantes
    tipos_acidentes = {
        "ABALROAMENTO": "#E151AF",     # Pink
        "ATROPELAMENTO": "#EDED12",    # Amarelo
        "CAPOTAGEM": "#88CDB2",        # Verde hortelã
        "CHOQUE": "#964B00",           # Marrom
        "COLISÃO": "#F3161E",          # Vermelho vivo
        "EVENTUAL": "#7026d9",         # Violeta
        "INCÊNDIO": "#FB8500",         # Laranja
        "NAO CADASTRADO": "#6C757D",   # Cinza neutro
        "OUTROS": "#023E8A",           # Azul marinho
        "QUEDA": "#69A4F1",            # Azul médio
        "TOMBAMENTO": "#00A385"        # Verde telha
    }
    
    ### Seleção dos dados do dataset a serem usados nas análises
    df_selected = df_2026_S1[["feridos", "fatais", "auto", "moto", "tipo_acid", "noite_dia"]]
    
    return df_2026_S1, df_selected, tipos_veiculos, tipos_acidentes


# Busca o dataset filtrado, tipos de veículos e tipos de acidentes

df_filtered, df_selected, tipos_veiculos, tipos_acidentes = carregar_dados()


### Totais de acidentes

total_acidentes_dataset = len(df_filtered)



# Inicializa as estruturas de visualização (executado apenas 1 vez)

#if "dados_visualizacao" not in st.session_state:
#    st.session_state["dados_visualizacao"] = {
#        "V1": {},
#        "V2": {},
#        "V3": {},
#        "V4": {}
#    }
#
#if "old_opcao_visualizacao" not in st.session_state:
#    st.session_state["old_opcao_visualizacao"] = "."


### Seleção dos dados do dataset a serem usados nas visualizações

#df_selected = df_filtered

# Exibe número de acidentes filtrados no painel à esquerda

#with st.sidebar:
#
#    # Número de acidentes filtrados
#    st.info(
#        f"Total de acidentes filtrados: {total_acidentes_selecionados:,.0f}"
#    )
#
#    # Divisor
#    st.divider()
#
#
#if df_selected.empty:
#    st.info("Não há ocorrências de acidentes para os filtros selecionados.")
#    st.stop()



### Área central do dashboard

with st.container(border=True, height="stretch", vertical_alignment="center"):

    ## Página Gráficos do dashboard

    if pagina_selecionada == "Gráficos":

        # Título primeira seção
        
        st.markdown("### Estatísticas gerais para os dados")
        
        
        ## Gráfico 1 do dashboard - Histograma vertical do total de acidentes por mês com linha de tendência geral
        
        df_grafico = df_filtered.copy()
        df_grafico['ano_mes'] = df_grafico['data'].dt.strftime('%Y-%m')

        # Agrupa por 'ano_mes' e conta os acidentes
        acidentes_por_mes = df_grafico.groupby('ano_mes').size().reset_index(name='total_acidentes')
        acidentes_por_mes = acidentes_por_mes.sort_values(by='ano_mes').reset_index(drop=True)

        if not acidentes_por_mes.empty:
            # Escala contínua: Verde (menor) -> Amarelo (médio) -> Vermelho (maior)
            escala_cores = [
                [0.0, '#27ae60'],   # Verde
                [0.5, '#f1c40f'],   # Amarelo
                [1.0, '#F3161E']    # Vermelho
            ]

            fig = go.Figure()

            # Barras do total de acidentes por mês com gradiente de cores
            fig.add_trace(
                go.Bar(
                    x=acidentes_por_mes['ano_mes'],
                    y=acidentes_por_mes['total_acidentes'],
                    name='Total de Acidentes',
                    marker=dict(
                        color=acidentes_por_mes['total_acidentes'],
                        colorscale=escala_cores,
                        showscale=False
                    ),
                    hovertemplate="<b>Ano-Mês:</b> %{x}<br><b>Total de Acidentes:</b> %{y:,.0f}<extra></extra>"
                )
            )

            # Cálculo e inclusão da Linha de Tendência Geral (Regressão Linear)
            if len(acidentes_por_mes) >= 2:
                x_indices = np.arange(len(acidentes_por_mes))
                y_valores = acidentes_por_mes['total_acidentes'].values
                inclinacao, intercepto = np.polyfit(x_indices, y_valores, 1)
                linha_tendencia = inclinacao * x_indices + intercepto

                # Linha pontilhada/tracejada de tendência
                fig.add_trace(
                    go.Scatter(
                        x=acidentes_por_mes['ano_mes'],
                        y=linha_tendencia,
                        mode='lines',
                        name='Tendência Geral',
                        line=dict(color='gray', dash='dash', width=2.5),
                        hovertemplate="<b>Tendência Geral:</b> %{y:,.1f}<extra></extra>"
                    )
                )

            fig.update_layout(
                title="Total de acidentes por mês",
                xaxis_title="Ano-Mês",
                yaxis_title="Número Total de Acidentes",
                xaxis=dict(tickangle=-90),
                yaxis=dict(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )

            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Não há dados mensais suficientes para exibir o gráfico.")
    


        ## Gráfico 2 do dashboard - Gráfico de linhas com evolução mensal por tipo de acidente

        df_linhas = df_filtered.copy()
        if not df_linhas.empty and "tipo_acid" in df_linhas.columns:
            df_linhas["ano_mes"] = df_linhas["data"].dt.strftime("%Y-%m")

            # Cria tabela cruzada agrupando ano/mês e tipo de acidente
            tabela_tipo_mes = pd.crosstab(
                df_linhas["ano_mes"],
                df_linhas["tipo_acid"]
            ).sort_index()

            # Ordena os tipos de acidente pelo total de ocorrências (decrescente na legenda)
            tipos_ordenados = (
                tabela_tipo_mes.sum()
                .sort_values(ascending=False)
                .index.tolist()
            )

            fig_linhas = go.Figure()

            for tipo in tipos_ordenados:
                cor = tipos_acidentes.get(tipo, "#6C757D")
                fig_linhas.add_trace(
                    go.Scatter(
                        x=tabela_tipo_mes.index,
                        y=tabela_tipo_mes[tipo],
                        mode="lines+markers",
                        name=tipo,
                        line=dict(color=cor, width=2.5),
                        marker=dict(size=4, color=cor),
                        hovertemplate=f"<b>Tipo:</b> {tipo}<br>"
                                      "<b>Ano-Mês:</b> %{x}<br>"
                                      "<b>Acidentes:</b> %{y:,.0f}<extra></extra>"
                    )
                )

            fig_linhas.update_layout(
                title="Evolução mensal de acidentes por tipo",
                xaxis_title="Ano-Mês",
                yaxis_title="Número de Acidentes",
                xaxis=dict(tickangle=-90),
                yaxis=dict(showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)"),
                legend=dict(
                    title=dict(text="Tipo de Acidente"),
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                height=500
            )

            st.plotly_chart(fig_linhas, width="stretch")
        else:
            st.info("Não há dados de tipos de acidentes para os filtros selecionados.")

       
       
        ## Gráfico 3 do dashboard - Histograma horizontal com o número de acidentes para cada tipo de veículo para o período selecionado
        
        # Filtro/máscara para acidentes sérios (feridos graves ou fatais >= 1)
        colunas_serios = [c for c in ["feridos_gr", "fatais"] if c in df_filtered.columns]
        if colunas_serios:
            mascara_serios = df_filtered[colunas_serios].ge(1).any(axis=1)
        else:
            mascara_serios = pd.Series(False, index=df_filtered.index)
        
        # Contagem de acidentes e acidentes sérios por tipo de veículo
        linhas_veiculos = []
        for tipo, colunas in tipos_veiculos.items():
            mascara_veiculo = df_filtered[colunas].ge(1).any(axis=1)
            total = mascara_veiculo.sum()
            serios = (mascara_veiculo & mascara_serios).sum()
            pct_serios = (serios / total * 100) if total > 0 else 0.0
            
            # Formata o texto exibido ao lado de cada barra: "total (X.X% sérios)"
            rotulo_texto = f"{total:,.0f} ({pct_serios:.1f}% sérios)"
            
            linhas_veiculos.append({
                "tipo_veiculo": tipo,
                "total_acidentes": total,
                "acidentes_serios": serios,
                "pct_serios": pct_serios,
                "rotulo_texto": rotulo_texto
            })
        
        dados_veiculos = pd.DataFrame(linhas_veiculos)
        
        # Ordena do maior para o menor
        dados_veiculos = dados_veiculos.sort_values(
            by="total_acidentes",
            ascending=False
        )
        
        # Barras
        barras = (
            alt.Chart(dados_veiculos)
            .mark_bar()
            .encode(
                x=alt.X(
                    "total_acidentes:Q",
                    title="Número de Acidentes",
                    scale=alt.Scale(
                        domain=[
                            0,
                            dados_veiculos["total_acidentes"].max() * 1.25  # Aumentado para acomodar o texto
                        ]
                    )
                ),
                y=alt.Y(
                    "tipo_veiculo:N",
                    title="Tipo de Veículo",
                    sort="-x"
                ),
                color=alt.Color(
                    "tipo_veiculo:N",
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("tipo_veiculo:N", title="Tipo de Veículo"),
                    alt.Tooltip("total_acidentes:Q", title="Total de Acidentes", format=",.0f"),
                    alt.Tooltip("acidentes_serios:Q", title="Acidentes Sérios", format=",.0f"),
                    alt.Tooltip("pct_serios:Q", title="% Sérios", format=".1f")
                ]
            )
        )
        
        # Rótulos com a quantidade e a porcentagem de acidentes sérios
        texto = (
            alt.Chart(dados_veiculos)
            .mark_text(
                align="left",
                baseline="middle",
                dx=5,
                color="black"
            )
            .encode(
                x="total_acidentes:Q",
                y=alt.Y("tipo_veiculo:N", sort="-x"),
                text="rotulo_texto:N"
            )
        )
        
        histograma_acidentes_tipo_veiculos = (
            (barras + texto)
            .properties(
                height=400,
                title="Acidentes por tipo de veículo (% de sérios entre parênteses)"
            )
        )
        
        st.altair_chart(histograma_acidentes_tipo_veiculos, width="stretch")
        


        ## Gráfico 5 do dashboard - Heatmap com os tipos de veículos e tipos de acidentes

        # Tipos de acidente existentes no conjunto filtrado
        tipos_acidente = sorted(df_filtered["tipo_acid"].dropna().unique())
        
        # Inicializa matriz de contagem
        heatmap_data = pd.DataFrame(
            index=tipos_acidente,
            columns=tipos_veiculos.keys(),
            data=0
        )
                    
        # Calcula as contagens
        for nome_veiculo, colunas in tipos_veiculos.items():

            mask = df_filtered[colunas].eq(1).any(axis=1)
            
            contagem = (
                df_filtered[mask]
                .groupby("tipo_acid")
                .size()
            )
            
            for tipo_acid, qtd in contagem.items():
                heatmap_data.loc[tipo_acid, nome_veiculo] = qtd
        
        # Adiciona totais por linha e por coluna
        heatmap_data["Total"] = heatmap_data.sum(axis=1)
        heatmap_data.loc["Total"] = heatmap_data.sum(axis=0)

        # Matriz de cores: cópia convertida para float definindo toda a linha e coluna "Total" como NaN
        # para que o Plotly não considere os valores de total no cálculo e na coloração do heatmap
        matriz_cores = heatmap_data.astype(float).copy()
        matriz_cores.loc["Total", :] = np.nan
        matriz_cores.loc[:, "Total"] = np.nan

        # Cria heatmap com a mesma escala ("YlOrRd") e zmin=0
        heatmap_veiculos_tipo_acidente = px.imshow(
            matriz_cores,
            color_continuous_scale="YlOrRd",
            zmin=0,
            aspect="auto",
            labels=dict(
                x="Tipo de veículo",
                y="Tipo de acidente",
                color="Nº de acidentes"
            ),
            text_auto=False,
            title="Heatmap de acidentes por tipo de veículo"
        )
        
        # Atualiza os textos numéricos exibidos nas células usando a matriz original (com os totais visíveis)
        heatmap_veiculos_tipo_acidente.update_traces(
            text=heatmap_data.values,
            texttemplate="%{text:.0f}",
            hovertemplate=
                "Veículo: %{x}<br>"
                "Acidente: %{y}<br>"
                "Ocorrências: %{text}<extra></extra>"
        )
        
        heatmap_veiculos_tipo_acidente.update_layout(
            xaxis_title="Tipo de veículo",
            yaxis_title="Tipo de acidente",
            height=max(500, len(tipos_acidente) * 30)
        )
        
        st.plotly_chart(heatmap_veiculos_tipo_acidente, width = 'stretch')



        ## Gráfico 6 do dashboard - Heatmap de acidentes por dia da semana x hora do dia

        # Filtra os dados válidos de dia da semana e hora
        df_heatmap_hora = df_filtered[
            df_filtered["dia_sem"].notna() & 
            df_filtered["hora"].notna()
        ].copy()

        if not df_heatmap_hora.empty:
            # Extrai a hora (00 a 23) em formato legível de 2 dígitos (ex: "00h", "01h", ...)
            df_heatmap_hora["hora_dia"] = df_heatmap_hora["hora"].dt.strftime("%Hh")

            # Garante a ordem correta dos dias da semana
            ordem_dias = [
                "DOMINGO", "SEGUNDA-FEIRA", "TERÇA-FEIRA",
                "QUARTA-FEIRA", "QUINTA-FEIRA", "SEXTA-FEIRA", "SÁBADO"
            ]

            # Cria a tabela cruzada (crosstab)
            matriz_dia_hora = pd.crosstab(
                df_heatmap_hora["dia_sem"],
                df_heatmap_hora["hora_dia"]
            )

            # Reordena os dias da semana que existem nos dados filtrados
            dias_presentes = [d for d in ordem_dias if d in matriz_dia_hora.index]
            matriz_dia_hora = matriz_dia_hora.reindex(dias_presentes)

            # Reordena as colunas de horas de 00h a 23h
            horas_ordenadas = [f"{h:02d}h" for h in range(24)]
            horas_presentes = [h for h in horas_ordenadas if h in matriz_dia_hora.columns]
            matriz_dia_hora = matriz_dia_hora.reindex(columns=horas_presentes, fill_value=0)

            # Adiciona o total ao final de cada linha e no final de cada coluna
            matriz_dia_hora["Total"] = matriz_dia_hora.sum(axis=1)
            matriz_dia_hora.loc["Total"] = matriz_dia_hora.sum(axis=0)

            # Matriz de cores: cópia da matriz original definindo a célula "Total x Total" como NaN
            # para que o Plotly não considere o valor acumulado no cálculo da escala de cores (color_continuous_scale)
            matriz_cores = matriz_dia_hora.astype(float).copy()
            matriz_cores.loc["Total", :] = np.nan
            matriz_cores.loc[:, "Total"] = np.nan

            # Cria o Heatmap interativo utilizando a escala de cores 'YlOrRd' (Amarelo, Laranja e Vermelho)
            fig_heatmap_dia_hora = px.imshow(
                matriz_cores,
                color_continuous_scale="YlOrRd",  # Transição de amarelo fraco -> laranja -> vermelho
                zmin=0,                            # Garante que o valor mínimo 0 pegue o amarelo mais claro
                aspect="auto",
                labels=dict(
                    x="Hora do Dia",
                    y="Dia da Semana",
                    color="Nº de Acidentes"
                ),
                text_auto=False, # Utilizaremos o parâmetro text explicitamente com a matriz completa
                title="Distribuição de acidentes por dia da semana e hora do dia"
            )

            # Atualiza os textos numéricos exibidos nas células usando a matriz original (garantindo que o Total Geral apareça como número)
            fig_heatmap_dia_hora.update_traces(
                text=matriz_dia_hora.values,
                texttemplate="%{text:.0f}",
                hovertemplate=
                    "Dia: %{y}<br>"
                    "Hora: %{x}<br>"
                    "Acidentes: %{text}<extra></extra>"
            )

            fig_heatmap_dia_hora.update_layout(
                xaxis_title="Hora do Dia",
                yaxis_title="Dia da Semana",
                height=500
            )

            st.plotly_chart(fig_heatmap_dia_hora, width='stretch')



        # Título segunda seção

        st.markdown("### Estatísticas de localização para os dados")

        
        
        ## Gráfico 7 do dashboard - Histograma horizontal com os logradouros com maior ocorrência de acidentes para o período selecionado
    
        # Contagem de acidentes por logradouro (log1) no dataset filtrado
        acidentes_por_logradouro = df_filtered['log1'].value_counts().reset_index()
        acidentes_por_logradouro.columns = ['logradouro', 'total_acidentes']
        
        # Obtenção dos 20 logradouros com maior número de ocorrências
        top_20_logradouros = acidentes_por_logradouro.head(20)
        
        # Barras
        barras = (
            alt.Chart(top_20_logradouros)
            .mark_bar()
            .encode(
                x=alt.X(
                    "total_acidentes:Q",
                    title="Número de Acidentes",
                    scale=alt.Scale(
                        domain=[
                            0,
                            top_20_logradouros["total_acidentes"].max() * 1.15
                            if not top_20_logradouros.empty else 10
                        ]
                    )
                ),
                y=alt.Y(
                    "logradouro:N",
                    title="Logradouro",
                    sort="-x",
                    axis=alt.Axis(
                        labelLimit=0,        # Evita truncar o nome das ruas
                        labelOverlap=False   # Garante a exibição de todos os rótulos do eixo Y
                    )
                ),
                color=alt.Color(
                    "total_acidentes:Q",
                    scale=alt.Scale(scheme="reds"),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip(
                        "logradouro:N",
                        title="Logradouro:"
                    ),
                    alt.Tooltip(
                        "total_acidentes:Q",
                        title="Acidentes:",
                        format=",.0f"
                    ),
                ]
            )
        )
        
        # Texto com a quantidade no final da barra
        texto = (
            alt.Chart(top_20_logradouros)
            .mark_text(
                align="left",
                baseline="middle",
                dx=5,
                color="black"
            )
            .encode(
                x="total_acidentes:Q",
                y=alt.Y("logradouro:N", sort="-x"),
                text=alt.Text("total_acidentes:Q", format=",.0f")
            )
        )
        
        histograma_acidentes_por_logradouros = (
            (barras + texto)
            .properties(
                height=800,
                title="Logradouros com maior número de acidentes para os dados"
            )
        )
        
        st.altair_chart(histograma_acidentes_por_logradouros, width="stretch")



        ## Gráfico 8 do dashboard - Histograma horizontal com os cruzamentos/esquinas com maior ocorrência de acidentes

        # Filtra apenas registros que possuem log1 e log2 preenchidos (que caracterizam um cruzamento/esquina)
        df_cruzamentos = df_filtered[
            df_filtered['log1'].notna() & 
            df_filtered['log2'].notna() & 
            (df_filtered['log1'].str.strip() != "") & 
            (df_filtered['log2'].str.strip() != "")
        ].copy()

        # Função para padronizar e ordenar o par de ruas do cruzamento
        def normalizar_cruzamento(row):
            rua1 = str(row['log1']).strip()
            rua2 = str(row['log2']).strip()
            # Ordena alfabeticamente para que (Rua A, Rua B) e (Rua B, Rua A) fiquem iguais
            esquina = sorted([rua1, rua2])
            return f"{esquina[0]} x {esquina[1]}"

        # Aplica a ordenação para formar o identificador único do cruzamento
        df_cruzamentos['cruzamento'] = df_cruzamentos.apply(normalizar_cruzamento, axis=1)

        # Contagem de acidentes por cruzamento
        acidentes_por_cruzamento = df_cruzamentos['cruzamento'].value_counts().reset_index()
        acidentes_por_cruzamento.columns = ['cruzamento', 'total_acidentes']

        # Obtenção dos 20 cruzamentos com maior número de ocorrências
        top_20_cruzamentos = acidentes_por_cruzamento.head(20)

        # Barras
        barras_cruzamento = (
            alt.Chart(top_20_cruzamentos)
            .mark_bar()
            .encode(
                x=alt.X(
                    "total_acidentes:Q",
                    title="Número de Acidentes",
                    scale=alt.Scale(
                        domain=[
                            0,
                            top_20_cruzamentos["total_acidentes"].max() * 1.15
                            if not top_20_cruzamentos.empty else 10
                        ]
                    )
                ),
                y=alt.Y(
                    "cruzamento:N",
                    title=None,          # Remove o título do eixo Y para não sobrepor com os nomes
                    sort="-x",
                    axis=alt.Axis(
                        labelLimit=0,        # Evita truncar o nome dos cruzamentos
                        labelOverlap=False   # Exibe todos os rótulos no eixo Y
                    )
                ),

                color=alt.Color(
                    "total_acidentes:Q",
                    scale=alt.Scale(scheme="reds"),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip(
                        "cruzamento:N",
                        title="Cruzamento:"
                    ),
                    alt.Tooltip(
                        "total_acidentes:Q",
                        title="Acidentes:",
                        format=",.0f"
                    ),
                ]
            )
        )
        
        # Texto com a quantidade no final de cada barra
        texto_cruzamento = (
            alt.Chart(top_20_cruzamentos)
            .mark_text(
                align="left",
                baseline="middle",
                dx=5,
                color="black"
            )
            .encode(
                x="total_acidentes:Q",
                y=alt.Y("cruzamento:N", sort="-x"),
                text=alt.Text("total_acidentes:Q", format=",.0f")
            )
        )
        
        histograma_acidentes_por_cruzamento = (
            (barras_cruzamento + texto_cruzamento)
            .properties(
                height=700, # Mantém a altura ampla para exibir todos os rótulos sem sobreposição
                title="Cruzamentos com maior número de acidentes para os dados"
            )
        )

        st.altair_chart(histograma_acidentes_por_cruzamento, width="stretch")
        
        
        
        ## Gráfico 9 do dashboard - Gráfico de rosca para comparar acidentes ao longo da via com acidentes nos cruzamentos

        # Calcula as contagens
        no_log2_count = df_filtered["log2"].isnull().sum()
        with_log2_count = df_filtered["log2"].notnull().sum()
        
        # Prepara o DataFrame para o Plotly
        df_pie = pd.DataFrame(
            {
                "Tipo de Local": ["Extensão da via", "Cruzamento"],
                "Quantidade": [no_log2_count, with_log2_count],
            }
        )
        
        # Cria o gráfico de rosca (hole=0.4 converte a pizza em rosca)
        fig = px.pie(
            df_pie,
            values="Quantidade",
            names="Tipo de Local",
            title="Acidentes por tipo de local (Extensão da via vs. Cruzamento)",
            color="Tipo de Local",
            color_discrete_map={
                "Extensão da via": "#7ad151",
                "Cruzamento": "#2a788e",
            },
            hole=0.45,  # Define o furo central da rosca
        )
        
        # Rótulos diretos no gráfico + efeito de contorno/sombra nas fatias
        fig.update_traces(
            textinfo="label+percent+value",
            texttemplate="<b>%{label}</b><br>%{percent:.1%}<br>(%{value:,.0f})",
            textposition="inside",  # Exibe os rótulos diretamente no corpo do gráfico
            insidetextorientation="horizontal",
            textfont=dict(size=13, color="white"),
            marker=dict(
                line=dict(color="rgba(0, 0, 0, 0.35)", width=2.5)  # Borda de contraste estilo sombra
            )
        )
        
        # Configuração de iluminação/sombra adicional nas fatias
        fig.update_layout(
            title_x=0.0,       # Alinha totalmente à esquerda do gráfico
            showlegend=False,  # Oculta a legenda já que os nomes estão direto nas fatias
            margin=dict(t=60, b=30, l=20, r=20),
        )
        
        # Formatação do gráfico (rótulos com porcentagem e valor absoluto)
        fig.update_traces(
            textinfo="percent+value",
            texttemplate="%{percent:.1%}<br>(%{value:,.0f})",
            textfont_size=13,
            marker=dict(line=dict(color="#000000", width=1)),
        )
        
        fig.update_layout(
            title_x=0.5,  # Centraliza o título
            legend_title_text="Local do Acidente",
            margin=dict(t=50, b=20, l=20, r=20),
        )
        
        # 7. Exibe o gráfico no Streamlit
        st.plotly_chart(fig, width='stretch')

    ####################################################################################################################################

    ## Página População do dashboard

    elif pagina_selecionada == "População":

        st.markdown("""
        
        ## Análise descritiva da população 🌐
        
        -----
        
        ### Dataset
        
        """)
        
        #st.markdown("## Análise descritiva da população")
        #st.markdown("---")
        #st.markdown("### Dataset")
    
        col1, col2 = st.columns(2)
        col1.metric("Total de acidentes no dataset:", f"{total_acidentes_dataset:,.0f}".replace(",", "."))
        
        st.markdown("""
        
        #### Informações sobre o conjunto de dados utilizado - cat_acidentes.csv
        
        |  |  |
        |---|---|
        | **Instituição ou responsável pelos dados** | Prefeitura Municipal de Porto Alegre |
        | **Página do conjunto de dados** | [Acessar página do conjunto](https://dadosabertos.poa.br/dataset/acidentes-de-transito-acidentes/resource/b56f8123-716a-4893-9348-23945f1ea1b9) |
        | **Link do arquivo** | [Download cat_acidentes.csv](https://dadosabertos.poa.br/dataset/d6cfbe48-ee1f-450f-87f5-9426f6a09328/resource/b56f8123-716a-4893-9348-23945f1ea1b9/download/cat_acidentes.csv) |
        | **Formato** | CSV |
        | **Data de acesso e versão/atualização** | Data e horário de acesso: 16/set/2026 às 20:40 - Não há informação de versão, dados foram atualizados em 1/set/2025 |
        | **Licença ou condições de uso** | Creative Commons Attribution |
        | **Restrições de acesso ou compartilhamento** | Não foi identificada |
        
        
        #### Dicionário de atributos

        | Atributo | Significado | Tipo semântico | Unidade ou categorias |
        |---|---|---|---|
        | predial1 | Número do logradouro onde ocorreu o acidente | Numérico | Identificador |
        | queda_arr | Informação se no acidente houve queda de algum veículo em arroio. | Numérico | 0 - Não, 1 - Sim |
        | data | Data em que ocorreu o acidente. | Numérico | Data |
        | feridos | Número de feridos no acidente. | Numérico | Unidades |
        | feridos_gr | Número de feridos graves no acidente. | Numérico | Unidades |
        | mortes | Contagem de vítimas fatais no momento do acidente. | Numérico | Unidades |
        | morte_post | Contagem de vítimas fatais posteriores ao momento do acidente e relacionadas ao mesmo. É considerado morte posterior a vítima que veio a óbito 30 dias após a data do acidente de trânsito. | Numérico | Unidades |
        | fatais | Somatório das vítimas fatais no momento do acidente e das vítimas posteriores relacionadas ao mesmo. | Numérico | Unidades |
        | auto | Número de veículos do tipo automóvel envolvidos no acidente. | Numérico | Unidades |
        | taxi | Número de táxis envolvidos no acidente | Numérico | Unidades |
        | lotacao | Número de veículos do tipo lotações envolvidas no acidente. | Numérico | Unidades |
        | onibus_urb | Número de ônibus urbanos envolvidos no acidente | Numérico | Unidades |
        | onibus_met | Número de ônibus metropolitanos envolvidos no acidente | Numérico | Unidades |
        | onibus_int | Número de ônibus interurbanos envolvidos no acidente | Numérico | Unidades |
        | caminhao | Número de veículos do tipo caminhão envolvidos no acidente. | Numérico | Unidades |
        | moto | Número de motocicletas envolvidas no acidente. | Numérico | Unidades |
        | carroca | Número de carroças envolvidas no acidente. | Numérico | Unidades |
        | bicicleta | Número de bicicletas envolvidas no acidente. | Numérico | Unidades |
        | outro | Número de outros veículos envolvidos no acidente | Numérico | Unidades |
        | cont_vit | Informação se o acidente possui ou não vítimas. | Numérico | 0 - Não, 1 - Sim |
        | ups | Unidade padrão de severidade: Peso atribuído aos tipos de acidentes de acordo com a gravidade dos danos causados. | Numérico | Unidades |
        | patinete | Número de patinetes envolvidos no acidente. | Numérico | Unidades |
        | idacidente | Número de identificação do acidente. | Numérico | Identificador |
        | longitude | Coordenada geográfica (eixo X) de localização do ponto onde ocorreu o acidente. | Geográfico | Coordenadas |
        | latitude | Coordenada geográfica (eixo Y) de localização do ponto onde ocorreu o acidente. | Geográfico | Coordenadas |
        | log1 | Nome do logradouro onde ocorreu o acidente. | Identificador | Texto |
        | log2 | Nome do logradouro que cruza o Logradouro no ponto onde ocorreu o acidente. | Identificador | Texto |
        | tipo_acid | Informação descritiva do tipo de acidente | Identificador | Texto |
        | dia_sem | Dia da semana em que ocorreu o acidente. | Temporal | Texto |
        | hora | Hora em que ocorreu o acidente. | Temporal | Texto |
        | noite_dia | Turno em que ocorreu o acidente. | Temporal | Texto |
        | regiao | Zona da cidade onde ocorreu o acidente de acordo com as divisões dos Postos de Controles Avançados (PCA) da fiscalização da EPTC. | Identificador | Texto |
        | consorcio | Consórcio responsável pelo (s) ônibus urbano (s) envolvido (s) no acidente. | Identificador | Texto |
        
        
        -----
        
        ### Variáveis selecionadas
        
        | Atributo | Significado | Tipo semântico | Unidade ou categorias |
        |---|---|---|---|
        | feridos | Número de feridos no acidente. | Numérico | Unidades |
        | fatais | Somatório das vítimas fatais no momento do acidente e das vítimas posteriores relacionadas ao mesmo. | Numérico | Unidades |
        | auto | Número de veículos do tipo automóvel envolvidos no acidente. | Numérico | Unidades |
        | moto | Número de motocicletas envolvidas no acidente. | Numérico | Unidades |
        | tipo_acid | Informação descritiva do tipo de acidente | Identificador | Texto |
        | noite_dia | Turno em que ocorreu o acidente. | Temporal | Texto |

        """)
                
        with st.expander("Estatísticas das variáveis selecionadas", expanded = True):
    
            # Gera as estatísticas e converte para string para evitar conflitos de tipos mistos no Arrow
            stats = df_selected.describe(include = "all").round(2)
            
            stats.rename(
                index = {
                    "count":  "Quantidade valores",
                    "unique": "Quant valores únicos",
                    "top":    "Valor mais frequente",
                    "freq":   "Freq valor mais freq",
                    "mean":   "Média",
                    "std":    "Desvio padrão",
                    "min":    "Valor mínimo",
                    "25%":    "Percentil 25%",
                    "50%":    "Mediana (Perc 50%)",
                    "75%":    "Percentil 75%",
                    "max":    "Valor máximo",
                },
                inplace = True
            )
            
            # Converte os valores para string (substituindo NaNs por vazios) para exibição limpa
            stats_exibicao = stats.fillna("").astype(str)
            
            # Altura calculada para acomodar o cabeçalho e todas as 11 linhas de estatísticas sem rolagem vertical
            st.dataframe(stats_exibicao, height=455, width='stretch')
        
        #with st.expander("Dados selecionados", expanded = True):
        #    # Garante que colunas de data/hora sejam convertidas para string formatada na visualização, se necessário
        #    df_exibicao = df_selected.copy()
        #    #for col in df_exibicao.select_dtypes(include=['datetime64', 'datetimetz']).columns:
        #    #    df_exibicao[col] = df_exibicao[col].astype(str)
        #
        #    st.dataframe(
        #        df_exibicao,
        #        hide_index = True,
        #        width = 'stretch'
        #    )
        
        
        
        # Cálculo das proporções de ocorrência (ocorrências > 0)
        total_acidentes = len(df_filtered)
        
        dados_prop = pd.DataFrame(
            [
                {
                    "variavel": "Acid com autos",
                    "proporcao": (df_filtered["auto"] > 0).mean() * 100,
                    "qtd": (df_filtered["auto"] > 0).sum(),
                },
                {
                    "variavel": "Acid com feridos",
                    "proporcao": (df_filtered["feridos"] > 0).mean() * 100,
                    "qtd": (df_filtered["feridos"] > 0).sum(),
                },
                {
                    "variavel": "Acid com motos",
                    "proporcao": (df_filtered["moto"] > 0).mean() * 100,
                    "qtd": (df_filtered["moto"] > 0).sum(),
                },
                {
                    "variavel": "Acid fatais",
                    "proporcao": (df_filtered["fatais"] > 0).mean() * 100,
                    "qtd": (df_filtered["fatais"] > 0).sum(),
                },
            ]
        ).sort_values(by="proporcao", ascending=False)
        
        # Formatação do rótulo
        dados_prop["rotulo"] = dados_prop.apply(
            lambda r: f"{r['proporcao']:.1f}% ({r['qtd']:,.0f})", axis=1
        )
        
        # Construção do gráfico de barras em Altair
        barras = (
            alt.Chart(dados_prop)
            .mark_bar()
            .encode(
                x=alt.X(
                    "proporcao:Q",
                    title="Porcentagem do total de acidentes (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                y=alt.Y("variavel:N", title="Variável", sort="-x"),
                color=alt.Color("variavel:N", legend=None),
                tooltip=[
                    alt.Tooltip("variavel:N", title="Variável"),
                    alt.Tooltip("proporcao:Q", title="Proporção do total(%)", format=".2f"),
                    alt.Tooltip("qtd:Q", title="Nº de acidentes", format=",.0f"),
                ],
            )
        )
        
        texto = (
            alt.Chart(dados_prop)
            .mark_text(align="left", baseline="middle", dx=5, color="black")
            .encode(
                x="proporcao:Q", y=alt.Y("variavel:N", sort="-x"), text="rotulo:N"
            )
        )
        
        grafico_proporcoes = (barras + texto).properties(
            height=300,
            title="Proporção de ocorrência das variáveis no total de acidentes",
        )
        
        st.altair_chart(grafico_proporcoes, width="stretch")
        
        
        
        ## Gráfico tipo_acid - Histograma horizontal com o número de acidentes para cada tipo de acidente para o período selecionado

        # Prepara uma cópia com o tipo de acidente preenchido
        df_temp = df_selected.copy()
        df_temp["tipo_acid_clean"] = df_temp["tipo_acid"].fillna("Não informado")

        # Agrupa para calcular total de acidentes por tipo
        acidentes_por_tipo = (
            df_temp.groupby("tipo_acid_clean")
            .size()
            .reset_index(name="total_acidentes")
            .rename(columns={"tipo_acid_clean": "tipo_acidente"})
        )

        # Total geral de acidentes no dataset filtrado
        total_geral_acidentes = len(df_selected)

        # Calcula a porcentagem de ocorrência em relação ao total de acidentes
        acidentes_por_tipo["pct_total"] = np.where(
            total_geral_acidentes > 0,
            (acidentes_por_tipo["total_acidentes"] / total_geral_acidentes) * 100,
            0.0
        )

        # Formata o rótulo de texto exibido ao lado de cada barra: "total (X.X% do total)"
        acidentes_por_tipo["rotulo_texto"] = acidentes_por_tipo.apply(
            lambda r: f"{r['pct_total']:.1f}% ({r['total_acidentes']:,.0f})",
            axis=1
        )

        # Ordena do maior para o menor total de acidentes
        acidentes_por_tipo = acidentes_por_tipo.sort_values(
            by="total_acidentes",
            ascending=False
        )

        # Lista dos tipos ordenados da maior para a menor frequência
        ordem_tipos = acidentes_por_tipo["tipo_acidente"].tolist()

        # Mapeia as cores respeitando exatamente a ordem calculada
        range_cores = [
            tipos_acidentes.get(tipo, "#333333") for tipo in ordem_tipos
        ]
        
        
        # Barras
        barras = (
            alt.Chart(acidentes_por_tipo)
            .mark_bar()
            .encode(
                x=alt.X(
                    "pct_total:Q",
                    title="Porcentagem do total de acidentes (%)",
                    scale=alt.Scale(
                        domain=[
                            0,
                            acidentes_por_tipo["pct_total"].max() * 1.30  # Folga estendida para acomodar o texto
                        ]
                    )
                ),
                y=alt.Y(
                    "tipo_acidente:N",
                    title="Tipo de Acidente",
                    sort=ordem_tipos
                ),
                color=alt.Color(
                    "tipo_acidente:N",
                    scale=alt.Scale(domain=ordem_tipos, range=range_cores),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("tipo_acidente:N", title="Tipo de acidente"),
                    alt.Tooltip("pct_total:Q", title="Proporção do total(%)", format=".1f"),
                    alt.Tooltip("total_acidentes:Q", title="Nº de acidentes", format=",.0f")
                ]
            )
        )

        # Rótulos com a quantidade e a porcentagem do total de acidentes
        texto = (
            alt.Chart(acidentes_por_tipo)
            .mark_text(
                align="left",
                baseline="middle",
                dx=5,  # distância em pixels após o final da barra
                color="black"
            )
            .encode(
                x="pct_total:Q",
                y=alt.Y("tipo_acidente:N", sort=ordem_tipos),
                text="rotulo_texto:N"
            )
        )

        histograma_acidentes_por_tipo = (
            (barras + texto)
            .properties(
                height=400,
                title="Variável tipo_acid - Proporção de ocorrência dos tipos de acidentes no total de acidentes"
            )
        )

        st.altair_chart(histograma_acidentes_por_tipo, width="stretch")
        
        
        
        ## Gráfico noite_dia - Histograma horizontal com a proporção de ocorrência por turno

        # Prepara uma cópia com o turno preenchido
        df_temp_turno = df_selected.copy()
        df_temp_turno["noite_dia_clean"] = df_temp_turno["noite_dia"].fillna("Não informado")

        # Agrupa para calcular total de acidentes por turno
        acidentes_por_turno = (
            df_temp_turno.groupby("noite_dia_clean")
            .size()
            .reset_index(name="total_acidentes")
            .rename(columns={"noite_dia_clean": "turno"})
        )

        # Total geral de acidentes no dataset filtrado
        total_geral_acidentes = len(df_selected)

        # Calcula a porcentagem de ocorrência em relação ao total de acidentes
        acidentes_por_turno["pct_total"] = np.where(
            total_geral_acidentes > 0,
            (acidentes_por_turno["total_acidentes"] / total_geral_acidentes) * 100,
            0.0
        )

        # Formata o rótulo de texto exibido ao lado de cada barra: "X.X% (total)"
        acidentes_por_turno["rotulo_texto"] = acidentes_por_turno.apply(
            lambda r: f"{r['pct_total']:.1f}% ({r['total_acidentes']:,.0f})",
            axis=1
        )

        # Ordena do maior para o menor total de acidentes
        acidentes_por_turno = acidentes_por_turno.sort_values(
            by="total_acidentes",
            ascending=False
        )

        # Lista dos turnos ordenados da maior para a menor frequência
        ordem_turnos = acidentes_por_turno["turno"].tolist()

        # Dicionário de cores para cada categoria
        cores_turno = {
            "DIA": "#4EA8DE",            # Azul claro
            "NOITE": "#001F54",          # Azul escuro
            "Não informado": "#6C757D"   # Cinza neutro
        }

        # Mapeia as cores na ordem exata dos dados ordenados
        range_cores_turno = [cores_turno.get(turno, "#6C757D") for turno in ordem_turnos]

        # Barras
        barras_turno = (
            alt.Chart(acidentes_por_turno)
            .mark_bar()
            .encode(
                x=alt.X(
                    "pct_total:Q",
                    title="Porcentagem do total de acidentes (%)",
                    scale=alt.Scale(
                        domain=[
                            0,
                            acidentes_por_turno["pct_total"].max() * 1.30
                        ]
                    )
                ),
                y=alt.Y(
                    "turno:N",
                    title="Turno (Noite / Dia)",
                    sort=ordem_turnos
                ),
                color=alt.Color(
                    "turno:N",
                    scale=alt.Scale(domain=ordem_turnos, range=range_cores_turno),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("turno:N", title="Turno"),
                    alt.Tooltip("pct_total:Q", title="Proporção do total(%)", format=".1f"),
                    alt.Tooltip("total_acidentes:Q", title="Nº de acidentes", format=",.0f")
                ]
            )
        )

        # Rótulos com a quantidade e a porcentagem do total de acidentes
        texto_turno = (
            alt.Chart(acidentes_por_turno)
            .mark_text(
                align="left",
                baseline="middle",
                dx=5,
                color="black"
            )
            .encode(
                x="pct_total:Q",
                y=alt.Y("turno:N", sort=ordem_turnos),
                text="rotulo_texto:N"
            )
        )

        histograma_acidentes_por_turno = (
            (barras_turno + texto_turno)
            .properties(
                height=250,
                title="Variável noite_dia - Proporção de ocorrência por turno no total de acidentes"
            )
        )

        st.altair_chart(histograma_acidentes_por_turno, width="stretch")
        
        
    ####################################################################################################################################

    ## Página Amostra do dashboard

    elif pagina_selecionada == "Amostra":

        st.markdown("""
        
        ## Análise descritiva da amostra 🧪
        
        -----
        
        ### 🧮 Calculadora de tamanho de amostra
        
        """)
        
        # ---------------------------------------------------------
        # Calculadora de Tamanho de Amostra (População Finita)
        # ---------------------------------------------------------

        # 1. Memória persistente entre navegações de página (chaves independentes de widgets)
        if "confianca_memoria" not in st.session_state:
            st.session_state["confianca_memoria"] = 95

        if "erro_memoria" not in st.session_state:
            st.session_state["erro_memoria"] = 5.0

        # Callback functions para atualizar a memória persistente sempre que o usuário alterar os valores
        def atualizar_confianca():
            st.session_state["confianca_memoria"] = st.session_state["temp_grau_confianca"]

        def atualizar_erro():
            st.session_state["erro_memoria"] = st.session_state["temp_margem_erro"]

        # Tabela de escores Z para níveis de confiança comuns
        z_scores = {
            80: 1.282,
            85: 1.440,
            90: 1.645,
            95: 1.960,
            99: 2.576
        }

        col_pop, col_conf, col_erro = st.columns(3)

        with col_pop:
            tamanho_populacao = st.number_input(
                "Tamanho da População (N)",
                min_value=1,
                value=int(len(df_selected)),
                step=1
            )

        with col_conf:
            opcoes_confianca = [80, 85, 90, 95, 99]
            val_conf_salvo = st.session_state["confianca_memoria"]
            idx_conf = opcoes_confianca.index(val_conf_salvo) if val_conf_salvo in opcoes_confianca else 3

            grau_confianca = st.selectbox(
                "Grau de Confiança (%)",
                options=opcoes_confianca,
                index=idx_conf,
                key="temp_grau_confianca",
                on_change=atualizar_confianca
            )

        with col_erro:
            margem_erro = st.number_input(
                "Margem de Erro (%)",
                min_value=0.1,
                max_value=50.0,
                value=float(st.session_state["erro_memoria"]),
                step=0.5,
                key="temp_margem_erro",
                on_change=atualizar_erro
            )

        # Cálculo da amostra usando os valores selecionados
        if tamanho_populacao > 0:
            Z = z_scores[grau_confianca]
            e = margem_erro / 100.0
            p = 0.5  # Proporção populacional conservadora (máxima variabilidade)

            # Fórmula para população finita
            num = tamanho_populacao * (Z ** 2) * p * (1 - p)
            den = ((e ** 2) * (tamanho_populacao - 1)) + ((Z ** 2) * p * (1 - p))
            
            n_amostra = int(np.ceil(num / den))
            n_amostra = min(n_amostra, tamanho_populacao)

            # Formatação dos valores
            n_amostra_fmt = f"{n_amostra:,.0f}".replace(",", ".")
            pct_val = (n_amostra / tamanho_populacao) * 100
            pct_fmt = f"{pct_val:.2f}".replace(".", ",")
            
            st.markdown(
                f"""
                <div style="font-family: inherit;">
                    <div style="font-size: 0.875rem; color: rgb(49, 51, 63); margin-bottom: 0.25rem;">
                        Tamanho da Amostra Necessária (n)
                    </div>
                    <div style="font-size: 2.25rem; font-weight: 600; color: rgb(49, 51, 63); line-height: 1.2;">
                        {n_amostra_fmt} 
                        <span style="font-size: 1.1rem; font-weight: 400; color: #555555; margin-left: 6px;">
                            ➔ {pct_fmt}% da população total
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
            
        # ---------------------------------------------------------
        # Seção de Amostragem Aleatória de Dados
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🎲 Gerador de Amostra Aleatória")

        # Botão para disparar/refazer a amostragem
        if st.button("Gerar Amostra Aleatória", type="primary"):
            # Garante que o tamanho da amostra não exceda o tamanho total do dataframe
            tamanho_real_amostra = min(n_amostra, len(df_selected))
            
            # Realiza a amostragem aleatória sem reposição mantendo o índice original
            df_amostrado = df_selected.sample(n=tamanho_real_amostra, random_state=None)
            
            # Armazena o resultado na sessão para persistência durante a navegação
            st.session_state["df_amostrado"] = df_amostrado

        # Exibe a amostragem se ela já tiver sido gerada
        if "df_amostrado" in st.session_state:
            df_amostrado = st.session_state["df_amostrado"]
            
            st.success(f"Amostra de {len(df_amostrado):,.0f} registros gerada com sucesso!".replace(",", "."))
            
            with st.expander("Visualização dos Primeiros e Últimos Itens Amostrados", expanded=True):
                # Concatena os 5 primeiros com os 5 últimos registros
                if len(df_amostrado) <= 10:
                    df_exibicao = df_amostrado.copy()
                else:
                    df_exibicao = pd.concat([df_amostrado.head(5), df_amostrado.tail(5)])
                
                # Reseta a chave do índice sem descartar o índice original da população
                df_exibicao_com_indice = df_exibicao.reset_index().rename(columns={"index": "Índice População"})
                
                # Renderiza no Streamlit sem ocultar a coluna do índice original
                st.dataframe(
                    df_exibicao_com_indice,
                    hide_index=True,
                    width="stretch"
                )    
            
            # ---------------------------------------------------------
            # Tabela Comparativa: População vs. Amostra
            # ---------------------------------------------------------
            st.markdown("#### 📊 Comparativo Estatístico: População vs. Amostra")

            # Descreve ambos os dataframes
            stats_pop = df_selected.describe(include="all").round(2)
            stats_amostra = df_amostrado.describe(include="all").round(2)

            # Mapeamento de nomes amigáveis para os índices estatísticos
            mapa_indices = {
                "count":  "Quantidade de valores",
                "unique": "Quant. valores únicos",
                "top":    "Valor mais frequente",
                "freq":   "Freq. valor mais frequente",
                "mean":   "Média",
                "std":    "Desvio padrão",
                "min":    "Valor mínimo",
                "25%":    "Percentil 25%",
                "50%":    "Mediana (Percentil 50%)",
                "75%":    "Percentil 75%",
                "max":    "Valor máximo",
            }

            stats_pop.rename(index=mapa_indices, inplace=True)
            stats_amostra.rename(index=mapa_indices, inplace=True)

            # Prepara a estrutura unificada para a tabela comparativa
            colunas_comparativas = {}

            for col in df_selected.columns:
                colunas_comparativas[f"{col} (População)"] = stats_pop[col].fillna("-").astype(str)
                colunas_comparativas[f"{col} (Amostra)"] = stats_amostra[col].fillna("-").astype(str)

            df_comparativo = pd.DataFrame(colunas_comparativas, index=stats_pop.index)

            with st.expander("Estatísticas Comparativas (População vs. Amostra)", expanded=True):
                st.dataframe(
                    df_comparativo,
                    height=455,
                    width="stretch"
                )
            
            
            # ---------------------------------------------------------
            # Gráfico Comparativo de Proporções: População vs. Amostra
            # ---------------------------------------------------------
            st.markdown("---")
            st.markdown("#### 📊 Comparativo de Proporção das Variáveis (População vs. Amostra)")

            # Parâmetros inferenciais para cálculo do erro da amostra
            N_total_pop = len(df_selected)
            n_total_amostra = len(df_amostrado)
            z_crit_grafico = z_scores.get(grau_confianca, 1.960)
            fcpf_grafico = np.sqrt((N_total_pop - n_total_amostra) / (N_total_pop - 1)) if N_total_pop > n_total_amostra else 1.0

            def calcular_margem_proporcao(p_pct, n_obs):
                if n_obs <= 0:
                    return 0.0
                p = p_pct / 100.0
                se = np.sqrt(max(0.0, p * (1.0 - p)) / n_obs) * fcpf_grafico
                return (z_crit_grafico * se) * 100.0

            # Cálculo das proporções para a População
            df_prop_pop = pd.DataFrame([
                {
                    "variavel": "Acid com autos",
                    "proporcao": (df_selected["auto"] > 0).mean() * 100,
                    "qtd": (df_selected["auto"] > 0).sum(),
                    "grupo": "População",
                    "erro_inf": (df_selected["auto"] > 0).mean() * 100,
                    "erro_sup": (df_selected["auto"] > 0).mean() * 100
                },
                {
                    "variavel": "Acid com feridos",
                    "proporcao": (df_selected["feridos"] > 0).mean() * 100,
                    "qtd": (df_selected["feridos"] > 0).sum(),
                    "grupo": "População",
                    "erro_inf": (df_selected["feridos"] > 0).mean() * 100,
                    "erro_sup": (df_selected["feridos"] > 0).mean() * 100
                },
                {
                    "variavel": "Acid com motos",
                    "proporcao": (df_selected["moto"] > 0).mean() * 100,
                    "qtd": (df_selected["moto"] > 0).sum(),
                    "grupo": "População",
                    "erro_inf": (df_selected["moto"] > 0).mean() * 100,
                    "erro_sup": (df_selected["moto"] > 0).mean() * 100
                },
                {
                    "variavel": "Acid fatais",
                    "proporcao": (df_selected["fatais"] > 0).mean() * 100,
                    "qtd": (df_selected["fatais"] > 0).sum(),
                    "grupo": "População",
                    "erro_inf": (df_selected["fatais"] > 0).mean() * 100,
                    "erro_sup": (df_selected["fatais"] > 0).mean() * 100
                },
            ])

            # Cálculo das proporções para a Amostra com cálculo da margem de erro
            amostra_vars = [
                ("Acid com autos", (df_amostrado["auto"] > 0).mean() * 100, (df_amostrado["auto"] > 0).sum()),
                ("Acid com feridos", (df_amostrado["feridos"] > 0).mean() * 100, (df_amostrado["feridos"] > 0).sum()),
                ("Acid com motos", (df_amostrado["moto"] > 0).mean() * 100, (df_amostrado["moto"] > 0).sum()),
                ("Acid fatais", (df_amostrado["fatais"] > 0).mean() * 100, (df_amostrado["fatais"] > 0).sum()),
            ]

            linhas_amostra_prop = []
            for var_nome, p_val, qtd_val in amostra_vars:
                margem_p = calcular_margem_proporcao(p_val, n_total_amostra)
                linhas_amostra_prop.append({
                    "variavel": var_nome,
                    "proporcao": p_val,
                    "qtd": qtd_val,
                    "grupo": "Amostra",
                    "erro_inf": max(0.0, p_val - margem_p),
                    "erro_sup": min(100.0, p_val + margem_p)
                })

            df_prop_amostra = pd.DataFrame(linhas_amostra_prop)

            # Unifica os dados para o gráfico
            df_prop_comp = pd.concat([df_prop_pop, df_prop_amostra])

            # Formatação do rótulo
            df_prop_comp["rotulo"] = df_prop_comp.apply(
                lambda r: f"{r['proporcao']:.1f}% ({r['qtd']:,.0f})", axis=1
            )

            ordem_vars = df_prop_pop.sort_values(by="proporcao", ascending=False)["variavel"].tolist()

            # Barras agrupadas com yOffset
            barras_comp = (
                alt.Chart(df_prop_comp)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "proporcao:Q",
                        title="Porcentagem do total de acidentes (%)",
                        scale=alt.Scale(domain=[0, min(100, df_prop_comp["erro_sup"].max() * 1.30)]),
                    ),
                    y=alt.Y("variavel:N", title="Variável", sort=ordem_vars),
                    yOffset="grupo:N",
                    color=alt.Color(
                        "grupo:N",
                        title="Conjunto",
                        scale=alt.Scale(domain=["População", "Amostra"], range=["#7fcdbb", "#de2d26"])
                    ),
                    tooltip=[
                        alt.Tooltip("grupo:N", title="Conjunto"),
                        alt.Tooltip("variavel:N", title="Variável"),
                        alt.Tooltip("proporcao:Q", title="Proporção (%)", format=".2f"),
                        alt.Tooltip("erro_inf:Q", title="Limite Inferior (%)", format=".2f"),
                        alt.Tooltip("erro_sup:Q", title="Limite Superior (%)", format=".2f"),
                        alt.Tooltip("qtd:Q", title="Nº de acidentes", format=",.0f"),
                    ],
                )
            )

            # Linha de margem de erro aplicada somente nas barras da Amostra
            df_erro_amostra = df_prop_comp[df_prop_comp["grupo"] == "Amostra"]
            
            linhas_erro_prop = (
                alt.Chart(df_erro_amostra)
                .mark_rule(color="#4a1412", strokeWidth=2.2)
                .encode(
                    x="erro_inf:Q",
                    x2="erro_sup:Q",
                    y=alt.Y("variavel:N", sort=ordem_vars),
                    yOffset="grupo:N"
                )
            )

            # Delimitador vertical (tique) na extremidade superior da margem de erro
            tique_erro_sup = (
                alt.Chart(df_erro_amostra)
                .mark_tick(color="#4a1412", thickness=2, size=12)
                .encode(
                    x="erro_sup:Q",
                    y=alt.Y("variavel:N", sort=ordem_vars),
                    yOffset="grupo:N"
                )
            )

            # Delimitador vertical (tique) na extremidade inferior da margem de erro
            tique_erro_inf = (
                alt.Chart(df_erro_amostra)
                .mark_tick(color="#4a1412", thickness=2, size=12)
                .encode(
                    x="erro_inf:Q",
                    y=alt.Y("variavel:N", sort=ordem_vars),
                    yOffset="grupo:N"
                )
            )

            # Rótulos com o percentual e contagem posicionados após a extensão máxima da barra/erro
            texto_comp = (
                alt.Chart(df_prop_comp)
                .mark_text(align="left", baseline="middle", dx=6, color="black")
                .encode(
                    x="erro_sup:Q",
                    y=alt.Y("variavel:N", sort=ordem_vars),
                    yOffset="grupo:N",
                    text="rotulo:N"
                )
            )

            grafico_prop_comparativo = (
                (barras_comp + linhas_erro_prop + tique_erro_sup + tique_erro_inf + texto_comp)
                .properties(
                    height=350,
                    title=f"Comparativo de Proporção das Variáveis (População vs. Amostra com Margem de Erro de {grau_confianca}% de Confiança)",
                )
            )

            st.altair_chart(grafico_prop_comparativo, width="stretch")
            
            
            # ---------------------------------------------------------
            # Gráfico Comparativo: tipo_acid (População vs. Amostra)
            # ---------------------------------------------------------
            st.markdown("---")
            st.markdown("#### 📊 Comparativo de Ocorrência por Tipo de Acidente (tipo_acid)")

            total_pop = len(df_selected)
            total_amostra = len(df_amostrado)

            # Prepara dados da População
            df_temp_pop = df_selected.copy()
            df_temp_pop["tipo_acid_clean"] = df_temp_pop["tipo_acid"].fillna("Não informado")
            
            acidentes_pop = (
                df_temp_pop.groupby("tipo_acid_clean")
                .size()
                .reset_index(name="total_acidentes")
                .rename(columns={"tipo_acid_clean": "tipo_acidente"})
            )
            acidentes_pop["pct_total"] = np.where(
                total_pop > 0, (acidentes_pop["total_acidentes"] / total_pop) * 100, 0.0
            )
            acidentes_pop["grupo"] = "População"
            acidentes_pop["erro_inf"] = acidentes_pop["pct_total"]
            acidentes_pop["erro_sup"] = acidentes_pop["pct_total"]

            # Prepara dados da Amostra
            df_temp_amostra = df_amostrado.copy()
            df_temp_amostra["tipo_acid_clean"] = df_temp_amostra["tipo_acid"].fillna("Não informado")
            
            acidentes_amostra = (
                df_temp_amostra.groupby("tipo_acid_clean")
                .size()
                .reset_index(name="total_acidentes")
                .rename(columns={"tipo_acid_clean": "tipo_acidente"})
            )
            acidentes_amostra["pct_total"] = np.where(
                total_amostra > 0, (acidentes_amostra["total_acidentes"] / total_amostra) * 100, 0.0
            )
            acidentes_amostra["grupo"] = "Amostra"
            
            # Margem de erro para cada categoria da amostra
            margens_tipo = acidentes_amostra["pct_total"].apply(lambda p: calcular_margem_proporcao(p, total_amostra))
            acidentes_amostra["erro_inf"] = (acidentes_amostra["pct_total"] - margens_tipo).clip(lower=0.0)
            acidentes_amostra["erro_sup"] = (acidentes_amostra["pct_total"] + margens_tipo).clip(upper=100.0)

            # Unifica todas as categorias
            todos_tipos = sorted(list(set(acidentes_pop["tipo_acidente"]).union(set(acidentes_amostra["tipo_acidente"]))))
            df_base = pd.DataFrame({"tipo_acidente": todos_tipos})
            
            acidentes_pop = pd.merge(df_base, acidentes_pop, on="tipo_acidente", how="left").fillna(
                {"total_acidentes": 0, "pct_total": 0.0, "grupo": "População", "erro_inf": 0.0, "erro_sup": 0.0}
            )
            acidentes_amostra = pd.merge(df_base, acidentes_amostra, on="tipo_acidente", how="left").fillna(
                {"total_acidentes": 0, "pct_total": 0.0, "grupo": "Amostra", "erro_inf": 0.0, "erro_sup": 0.0}
            )

            df_tipo_comp = pd.concat([acidentes_pop, acidentes_amostra])
            df_tipo_comp["rotulo_texto"] = df_tipo_comp.apply(
                lambda r: f"{r['pct_total']:.1f}% ({r['total_acidentes']:,.0f})", axis=1
            )

            ordem_tipos_comp = (
                acidentes_pop.sort_values(by="total_acidentes", ascending=False)["tipo_acidente"].tolist()
            )

            barras_tipo_comp = (
                alt.Chart(df_tipo_comp)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "pct_total:Q",
                        title="Porcentagem do total de acidentes (%)",
                        scale=alt.Scale(
                            domain=[0, df_tipo_comp["erro_sup"].max() * 1.30]
                        ),
                    ),
                    y=alt.Y(
                        "tipo_acidente:N",
                        title="Tipo de Acidente",
                        sort=ordem_tipos_comp
                    ),
                    yOffset="grupo:N",
                    color=alt.Color(
                        "grupo:N",
                        title="Conjunto",
                        scale=alt.Scale(domain=["População", "Amostra"], range=["#7fcdbb", "#de2d26"])
                    ),
                    tooltip=[
                        alt.Tooltip("grupo:N", title="Conjunto"),
                        alt.Tooltip("tipo_acidente:N", title="Tipo de acidente"),
                        alt.Tooltip("pct_total:Q", title="Proporção do total (%)", format=".1f"),
                        alt.Tooltip("erro_inf:Q", title="Limite Inferior (%)", format=".1f"),
                        alt.Tooltip("erro_sup:Q", title="Limite Superior (%)", format=".1f"),
                        alt.Tooltip("total_acidentes:Q", title="Nº de acidentes", format=",.0f")
                    ]
                )
            )

            df_tipo_erro_amostra = df_tipo_comp[df_tipo_comp["grupo"] == "Amostra"]

            linhas_erro_tipo = (
                alt.Chart(df_tipo_erro_amostra)
                .mark_rule(color="#4a1412", strokeWidth=2.0)
                .encode(
                    x="erro_inf:Q",
                    x2="erro_sup:Q",
                    y=alt.Y("tipo_acidente:N", sort=ordem_tipos_comp),
                    yOffset="grupo:N"
                )
            )

            tique_tipo_sup = (
                alt.Chart(df_tipo_erro_amostra)
                .mark_tick(color="#4a1412", thickness=2, size=10)
                .encode(
                    x="erro_sup:Q",
                    y=alt.Y("tipo_acidente:N", sort=ordem_tipos_comp),
                    yOffset="grupo:N"
                )
            )

            tique_tipo_inf = (
                alt.Chart(df_tipo_erro_amostra)
                .mark_tick(color="#4a1412", thickness=2, size=10)
                .encode(
                    x="erro_inf:Q",
                    y=alt.Y("tipo_acidente:N", sort=ordem_tipos_comp),
                    yOffset="grupo:N"
                )
            )

            texto_tipo_comp = (
                alt.Chart(df_tipo_comp)
                .mark_text(
                    align="left",
                    baseline="middle",
                    dx=6,
                    color="black"
                )
                .encode(
                    x="erro_sup:Q",
                    y=alt.Y("tipo_acidente:N", sort=ordem_tipos_comp),
                    yOffset="grupo:N",
                    text="rotulo_texto:N"
                )
            )

            histograma_tipo_comp = (
                (barras_tipo_comp + linhas_erro_tipo + tique_tipo_sup + tique_tipo_inf + texto_tipo_comp)
                .properties(
                    height=max(400, len(todos_tipos) * 45),
                    title=f"Variável tipo_acid - Comparativo de Proporção (População vs. Amostra com Margem de Erro de {grau_confianca}% de Confiança)"
                )
            )

            st.altair_chart(histograma_tipo_comp, width="stretch")


            # ---------------------------------------------------------
            # Gráfico Comparativo: noite_dia (População vs. Amostra)
            # ---------------------------------------------------------
            st.markdown("---")
            st.markdown("#### 📊 Comparativo de Ocorrência por Turno (noite_dia)")

            total_pop = len(df_selected)
            total_amostra = len(df_amostrado)

            # Prepara dados da População
            df_temp_turno_pop = df_selected.copy()
            df_temp_turno_pop["noite_dia_clean"] = df_temp_turno_pop["noite_dia"].fillna("Não informado")
            
            acidentes_turno_pop = (
                df_temp_turno_pop.groupby("noite_dia_clean")
                .size()
                .reset_index(name="total_acidentes")
                .rename(columns={"noite_dia_clean": "turno"})
            )
            acidentes_turno_pop["pct_total"] = np.where(
                total_pop > 0, (acidentes_turno_pop["total_acidentes"] / total_pop) * 100, 0.0
            )
            acidentes_turno_pop["grupo"] = "População"
            acidentes_turno_pop["erro_inf"] = acidentes_turno_pop["pct_total"]
            acidentes_turno_pop["erro_sup"] = acidentes_turno_pop["pct_total"]

            # Prepara dados da Amostra
            df_temp_turno_amostra = df_amostrado.copy()
            df_temp_turno_amostra["noite_dia_clean"] = df_temp_turno_amostra["noite_dia"].fillna("Não informado")
            
            acidentes_turno_amostra = (
                df_temp_turno_amostra.groupby("noite_dia_clean")
                .size()
                .reset_index(name="total_acidentes")
                .rename(columns={"noite_dia_clean": "turno"})
            )
            acidentes_turno_amostra["pct_total"] = np.where(
                total_amostra > 0, (acidentes_turno_amostra["total_acidentes"] / total_amostra) * 100, 0.0
            )
            acidentes_turno_amostra["grupo"] = "Amostra"

            margens_turno = acidentes_turno_amostra["pct_total"].apply(lambda p: calcular_margem_proporcao(p, total_amostra))
            acidentes_turno_amostra["erro_inf"] = (acidentes_turno_amostra["pct_total"] - margens_turno).clip(lower=0.0)
            acidentes_turno_amostra["erro_sup"] = (acidentes_turno_amostra["pct_total"] + margens_turno).clip(upper=100.0)

            # Unifica todas as categorias
            todos_turnos = sorted(list(set(acidentes_turno_pop["turno"]).union(set(acidentes_turno_amostra["turno"]))))
            df_base_turno = pd.DataFrame({"turno": todos_turnos})
            
            acidentes_turno_pop = pd.merge(df_base_turno, acidentes_turno_pop, on="turno", how="left").fillna(
                {"total_acidentes": 0, "pct_total": 0.0, "grupo": "População", "erro_inf": 0.0, "erro_sup": 0.0}
            )
            acidentes_turno_amostra = pd.merge(df_base_turno, acidentes_turno_amostra, on="turno", how="left").fillna(
                {"total_acidentes": 0, "pct_total": 0.0, "grupo": "Amostra", "erro_inf": 0.0, "erro_sup": 0.0}
            )

            df_turno_comp = pd.concat([acidentes_turno_pop, acidentes_turno_amostra])
            df_turno_comp["rotulo_texto"] = df_turno_comp.apply(
                lambda r: f"{r['pct_total']:.1f}% ({r['total_acidentes']:,.0f})", axis=1
            )

            ordem_turnos_comp = (
                acidentes_turno_pop.sort_values(by="total_acidentes", ascending=False)["turno"].tolist()
            )

            barras_turno_comp = (
                alt.Chart(df_turno_comp)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "pct_total:Q",
                        title="Porcentagem do total de acidentes (%)",
                        scale=alt.Scale(
                            domain=[0, df_turno_comp["erro_sup"].max() * 1.30]
                        ),
                    ),
                    y=alt.Y(
                        "turno:N",
                        title="Turno (Noite / Dia)",
                        sort=ordem_turnos_comp
                    ),
                    yOffset="grupo:N",
                    color=alt.Color(
                        "grupo:N",
                        title="Conjunto",
                        scale=alt.Scale(domain=["População", "Amostra"], range=["#7fcdbb", "#de2d26"])
                    ),
                    tooltip=[
                        alt.Tooltip("grupo:N", title="Conjunto"),
                        alt.Tooltip("turno:N", title="Turno"),
                        alt.Tooltip("pct_total:Q", title="Proporção do total (%)", format=".1f"),
                        alt.Tooltip("erro_inf:Q", title="Limite Inferior (%)", format=".1f"),
                        alt.Tooltip("erro_sup:Q", title="Limite Superior (%)", format=".1f"),
                        alt.Tooltip("total_acidentes:Q", title="Nº de acidentes", format=",.0f")
                    ]
                )
            )

            df_turno_erro_amostra = df_turno_comp[df_turno_comp["grupo"] == "Amostra"]

            linhas_erro_turno = (
                alt.Chart(df_turno_erro_amostra)
                .mark_rule(color="#4a1412", strokeWidth=2.0)
                .encode(
                    x="erro_inf:Q",
                    x2="erro_sup:Q",
                    y=alt.Y("turno:N", sort=ordem_turnos_comp),
                    yOffset="grupo:N"
                )
            )

            tique_turno_sup = (
                alt.Chart(df_turno_erro_amostra)
                .mark_tick(color="#4a1412", thickness=2, size=10)
                .encode(
                    x="erro_sup:Q",
                    y=alt.Y("turno:N", sort=ordem_turnos_comp),
                    yOffset="grupo:N"
                )
            )

            tique_turno_inf = (
                alt.Chart(df_turno_erro_amostra)
                .mark_tick(color="#4a1412", thickness=2, size=10)
                .encode(
                    x="erro_inf:Q",
                    y=alt.Y("turno:N", sort=ordem_turnos_comp),
                    yOffset="grupo:N"
                )
            )

            texto_turno_comp = (
                alt.Chart(df_turno_comp)
                .mark_text(
                    align="left",
                    baseline="middle",
                    dx=6,
                    color="black"
                )
                .encode(
                    x="erro_sup:Q",
                    y=alt.Y("turno:N", sort=ordem_turnos_comp),
                    yOffset="grupo:N",
                    text="rotulo_texto:N"
                )
            )

            histograma_turno_comp = (
                (barras_turno_comp + linhas_erro_turno + tique_turno_sup + tique_turno_inf + texto_turno_comp)
                .properties(
                    height=300,
                    title=f"Variável noite_dia - Comparativo de Proporção (População vs. Amostra com Margem de Erro de {grau_confianca}% de Confiança)"
                )
            )

            st.altair_chart(histograma_turno_comp, width="stretch")        



    ####################################################################################################################################

    ## Página Inferência 1 do dashboard

    elif pagina_selecionada == "Inferência 1":

        # ---------------------------------------------------------
        # Recuperação dos parâmetros selecionados na página Amostra
        # ---------------------------------------------------------
        z_scores = {
            80: 1.282,
            85: 1.440,
            90: 1.645,
            95: 1.960,
            99: 2.576
        }

        grau_conf_val = int(st.session_state.get("confianca_memoria", 95))
        margem_erro_val = float(st.session_state.get("erro_memoria", 5.0))
        z_critico = z_scores.get(grau_conf_val, 1.960)

        # Formatações para exibição
        conf_str = f"{grau_conf_val}%"
        z_crit_str = f"{z_critico:.2f}".replace(".", ",")
        erro_str = f"{margem_erro_val:.1f}%".replace(".", ",")
        p_ref = 0.02  # Valor de referência da hipótese nula (2%)
        p_ref_str = "2,00%"
        
        
        st.markdown(rf"""
        
        ## Inferência estatística e análise dos resultados 🔮 
        
        ---
        
        ### Inferência 1: Proporção de Acidentes Fatais
        
        * **Pergunta investigativa:**
          A proporção de acidentes fatais ocorridos em Porto Alegre é diferente de 2%, que é a proporção aproximada para as grandes cidades brasileiras?

        * **Variável investigada:**
          * $p_{{fatais}}$: proporção de acidentes com a variável `fatais > 0` em relação ao total.

        * **Hipóteses estatísticas:**
          * **Hipótese nula ($H_0$):** $p_{{fatais}} = 2\%$ ($p_0 = 0,02$)
          * **Hipótese alternativa ($H_1$):** $p_{{fatais}} \neq 2\%$ ($p_0 \neq 0,02$)

        * **Parâmetros Amostrais Adotados:** Grau de Confiança de **{conf_str}** ($Z_{{crítico}} = {z_crit_str}$) e Margem de Erro de **{erro_str}**.

        ---
        """)
        

        # ---------------------------------------------------------
        # 1. PARÂMETROS REAIS DA POPULAÇÃO (Censo / Ground Truth)
        # ---------------------------------------------------------
        N_pop = len(df_selected)
        k_fatais_pop = (df_selected["fatais"] > 0).sum()
        p_fatais_pop = k_fatais_pop / N_pop if N_pop > 0 else 0.0
        diff_pop_ref_pp = (p_fatais_pop - p_ref) * 100

        # ---------------------------------------------------------
        # 2. TESTE DE INFERÊNCIA NA AMOSTRA
        # ---------------------------------------------------------
        st.markdown("### 🧪 1. Inferência Estatística a partir da Amostra")

        if "df_amostrado" in st.session_state:
            df_amostra = st.session_state["df_amostrado"]
            n_obs = len(df_amostra)

            # Fator de Correção de População Finita (FCPF)
            fcpf = np.sqrt((N_pop - n_obs) / (N_pop - 1)) if N_pop > n_obs else 1.0

            # Proporção amostral estimada
            k_fatais_amostra = (df_amostra["fatais"] > 0).sum()
            p_fatais_hat = k_fatais_amostra / n_obs if n_obs > 0 else 0.0

            # Intervalo de Confiança via aproximação Normal com FCPF
            se_amostra = np.sqrt((p_fatais_hat * (1 - p_fatais_hat)) / n_obs) * fcpf if n_obs > 0 else 0.0
            ic_fatais_inf = max(0.0, p_fatais_hat - (z_critico * se_amostra))
            ic_fatais_sup = min(1.0, p_fatais_hat + (z_critico * se_amostra))

            # Teste de Hipótese Z para uma proporção sob H0 (usando p_0 = 0.02 no erro padrão)
            se_teste_h0 = np.sqrt((p_ref * (1 - p_ref)) / n_obs) * fcpf if n_obs > 0 else 1.0
            z_stat = (p_fatais_hat - p_ref) / se_teste_h0 if se_teste_h0 > 0 else 0.0
            rejeita_h0 = abs(z_stat) > z_critico

            p_fatais_str = f"{p_fatais_hat * 100:.2f}%".replace(".", ",")
            ic_fatais_str = f"[{ic_fatais_inf * 100:.2f}%, {ic_fatais_sup * 100:.2f}%]".replace(".", ",")
            contagem_fatais_str = f"({k_fatais_amostra:,}/{n_obs:,})".replace(",", ".")
            z_stat_str = f"{abs(z_stat):.2f}".replace(".", ",")

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.info(rf"**Estimativa Amostral: Acidentes Fatais ($\hat{{p}}_{{fatais}}$)**")
                st.markdown(
                    rf"""
                    <div style="font-size: 1.05rem; line-height: 1.8;">
                        • <b>Proporção na amostra:</b> 
                        <span style="font-size: 1.45rem; font-weight: 700; color: #1f77b4; margin-left: 4px;">
                            {p_fatais_str}
                        </span> 
                        <span style="font-size: 0.95rem; color: #555555;">{contagem_fatais_str}</span>
                        <br>
                        • <b>Intervalo de Confiança ({conf_str}):</b> 
                        <span style="font-size: 1.30rem; font-weight: 600; color: #2c3e50; margin-left: 4px;">
                            {ic_fatais_str}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_a2:
                st.info(r"**Hipótese de Teste de Referência ($H_0$)**")
                
                st.markdown(
                    rf"""
                    <div style="font-size: 1.05rem; line-height: 1.8;">
                        • <b>Valor de Referência (<i>p</i><sub>0</sub>):</b> 
                        <span style="font-size: 1.45rem; font-weight: 700; color: #7fcdbb; margin-left: 4px;">
                            {p_ref_str}
                        </span>
                        <br>
                        • <b>Estatística de Teste (|<i>Z</i>|):</b> 
                        <span style="font-size: 1.30rem; font-weight: 600; color: #2c3e50; margin-left: 4px;">
                            {z_stat_str}
                        </span> 
                        <span style="font-size: 0.95rem; color: #555555;">(Crítico: {z_crit_str})</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if rejeita_h0:
                rel_amostra = "superior" if p_fatais_hat > p_ref else "inferior"
                st.success(
                    rf"**Conclusão da Inferência Amostral: Rejeita-se $H_0$!** "
                    rf"Com base na amostra coletada ($n = {n_obs:,}$), a proporção de acidentes fatais (`{p_fatais_str}`) "
                    rf"é **estatisticamente {rel_amostra}** a 2,00% ao nível de {conf_str} de confiança (|Z| = {z_stat_str} > {z_crit_str}).".replace(",", ".")
                )
            else:
                st.warning(
                    rf"**Conclusão da Inferência Amostral: Não se rejeita $H_0$.** "
                    rf"Com base na amostra ($n = {n_obs:,}$), não há evidência estatística suficiente ao nível de {conf_str} para afirmar que a proporção de acidentes fatais seja diferente de 2,00% (|Z| = {z_stat_str} <= {z_crit_str}).".replace(",", ".")
                )
        else:
            st.info("💡 *Nenhuma amostra foi gerada até o momento. Acesse a página '🧪 Amostra' e gere uma amostra aleatória para visualizar o teste inferencial.*")

        # ---------------------------------------------------------
        # 3. PARÂMETROS REAIS DA POPULAÇÃO (Ground Truth)
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🌐 2. Parâmetros Reais da População (Censo / Ground Truth)")
        st.caption("Como estes dados compreendem a totalidade dos registros da população selecionada, os valores abaixo representam os parâmetros reais exatos ($p$), dispensando intervalos de confiança ou testes de hipótese probabilísticos.")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric(
                label="Proporção Real de Fatais (p_fatais)",
                value=f"{p_fatais_pop * 100:.2f}%".replace(".", ","),
                help=f"{k_fatais_pop:,} acidentes fatais de {N_pop:,}".replace(",", ".")
            )
        with col_p2:
            st.metric(
                label="Valor de Referência (p_0)",
                value=p_ref_str,
                help="Parâmetro hipotético sob teste (2%)."
            )
        with col_p3:
            st.metric(
                label="Diferença Real Populacional (p_fatais - 2%)",
                value=f"{diff_pop_ref_pp:.2f}".replace(".", ",") + " p.p.",
                help="Diferença determinística observada na população."
            )

        # ---------------------------------------------------------
        # 4. VALIDAÇÃO E INTERPRETAÇÃO DA INFERÊNCIA
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### ⚖️ 3. Avaliação e Validação da Inferência Amostral")

        if "df_amostrado" in st.session_state:
            fatais_coberto = ic_fatais_inf <= p_fatais_pop <= ic_fatais_sup
            ref_no_ic = ic_fatais_inf <= p_ref <= ic_fatais_sup

            df_grafico_val = pd.DataFrame([
                {
                    "categoria": "Acidentes Fatais",
                    "p_hat": p_fatais_hat * 100,
                    "ic_inf": ic_fatais_inf * 100,
                    "ic_sup": ic_fatais_sup * 100,
                    "p_real": p_fatais_pop * 100,
                    "p_ref": p_ref * 100,
                    "rotulo_amostra": f"{p_fatais_hat * 100:.2f}% (IC {conf_str}: [{ic_fatais_inf * 100:.2f}% - {ic_fatais_sup * 100:.2f}%])".replace(".", ",")
                }
            ])

            # Barra do IC da Amostra
            barras_amostra = (
                alt.Chart(df_grafico_val)
                .mark_rule(strokeWidth=4, color="#de2d26")
                .encode(
                    x=alt.X(
                        "ic_inf:Q",
                        title="Proporção (%)",
                        scale=alt.Scale(
                            domain=[
                                max(0, min(df_grafico_val["ic_inf"].min(), df_grafico_val["p_real"].min(), 1.0) * 0.80),
                                max(df_grafico_val["ic_sup"].max(), df_grafico_val["p_real"].max(), 2.5) * 1.25
                            ]
                        )
                    ),
                    x2="ic_sup:Q",
                    y=alt.Y("categoria:N", title=None)
                )
            )

            # Ponto da estimativa pontual da amostra (\hat{p})
            pontos_amostra = (
                alt.Chart(df_grafico_val)
                .mark_point(size=130, filled=True, color="#de2d26")
                .encode(
                    x="p_hat:Q",
                    y=alt.Y("categoria:N"),
                    tooltip=[
                        alt.Tooltip("categoria:N", title="Categoria"),
                        alt.Tooltip("p_hat:Q", title="Estimativa Amostral (%)", format=".2f"),
                        alt.Tooltip("ic_inf:Q", title=f"IC Inf ({conf_str})", format=".2f"),
                        alt.Tooltip("ic_sup:Q", title=f"IC Sup ({conf_str})", format=".2f")
                    ]
                )
            )

            # Traço do Parâmetro Real da População (Ground Truth)
            pontos_real = (
                alt.Chart(df_grafico_val)
                .mark_tick(size=25, thickness=4, color="#1b7837")
                .encode(
                    x="p_real:Q",
                    y=alt.Y("categoria:N"),
                    tooltip=[
                        alt.Tooltip("categoria:N", title="Categoria"),
                        alt.Tooltip("p_real:Q", title="Parâmetro Real População (%)", format=".2f")
                    ]
                )
            )

            # Linha vertical pontilhada indicando o limiar de teste H0 (2,00%)
            df_ref_linha = pd.DataFrame([{"ref": p_ref * 100}])
            linha_ref = (
                alt.Chart(df_ref_linha)
                .mark_rule(strokeDash=[6, 4], strokeWidth=2, color="#001F54")
                .encode(
                    x="ref:Q",
                    tooltip=[alt.Tooltip("ref:Q", title="Valor de Referência H0 (%)", format=".2f")]
                )
            )

            # Rótulo em texto com a proporção da amostra e o IC
            texto_amostra = (
                alt.Chart(df_grafico_val)
                .mark_text(align="left", baseline="middle", dx=12, color="#333333")
                .encode(
                    x="ic_sup:Q",
                    y=alt.Y("categoria:N"),
                    text="rotulo_amostra:N"
                )
            )

            grafico_validacao = (
                (barras_amostra + pontos_amostra + pontos_real + linha_ref + texto_amostra)
                .properties(
                    height=160,
                    title=f"Validação: IC {conf_str} da Amostra (Vermelho) vs. Parâmetro Real (Verde) vs. Referência H0 de 2% (Linha Azul)"
                )
            )

            st.altair_chart(grafico_validacao, width="stretch")

            # Tabela de validação
            df_tabela_validacao = pd.DataFrame([
                {
                    "Variável": "Acidentes Fatais",
                    "Amostra (p_hat)": f"{p_fatais_hat * 100:.2f}%".replace(".", ","),
                    f"IC {conf_str} da Amostra": f"[{ic_fatais_inf * 100:.2f}%, {ic_fatais_sup * 100:.2f}%]".replace(".", ","),
                    "Parâmetro Real População (p)": f"{p_fatais_pop * 100:.2f}%".replace(".", ","),
                    "O Parâmetro Real está no IC?": "✅ Sim" if fatais_coberto else "❌ Não",
                    "Valor 2% está no IC?": "Sim (H0 Aceita)" if ref_no_ic else "Não (H0 Rejeitada)"
                }
            ])

            st.dataframe(df_tabela_validacao, hide_index=True, width="stretch")

            # Conclusão metodológica dinâmica
            N_pop_txt = f"{N_pop}".replace(",", ".")
            p_fatais_pop_txt = f"{p_fatais_pop * 100:.2f}%".replace(".", ",")
            diff_real_txt = f"{abs(diff_pop_ref_pp):.2f}".replace(".", ",")
            rel_real = "superior" if p_fatais_pop > p_ref else "inferior"
            complemento_alfa = f"{100 - grau_conf_val}%"

            # Avalia se a decisão da amostra corresponde ao que de fato ocorre na população
            pop_rejeitaria_h0 = p_fatais_pop != p_ref
            concordancia = (rejeita_h0 == pop_rejeitaria_h0)

            if concordancia and rejeita_h0:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica:** A amostragem aleatória com nível de confiança de {conf_str} demonstrou alta eficácia, "
                    f"permitindo detectar que a proporção real de acidentes fatais diverge de 2,00% (|Z| = {z_stat_str} > {z_crit_str}) "
                    f"sem a necessidade de analisar a totalidade dos dados."
                )
            elif not rejeita_h0 and abs(diff_pop_ref_pp) > 0.3:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica (Alerta de Poder Amostral):** A amostra não rejeitou a hipótese nula $H_0$ (|Z| = {z_stat_str} <= {z_crit_str}), "
                    f"enquanto na população completa a proporção é de fato {rel_real} a 2,00% por {diff_real_txt} p.p. "
                    f"Esse comportamento ilustra a ocorrência do **Erro Tipo II (falso negativo)**, comum quando o evento possui baixa prevalência "
                    f"(acidentes fatais) e o tamanho amostral é insuficiente para isolar pequenas diferenças percentuais frente à margem de erro ({erro_str})."
                )
            else:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica:** A inferência amostral indicou proximidade entre a proporção observada e o limiar de 2,00% (|Z| = {z_stat_str} <= {z_crit_str}), "
                    f"mostrando consistência estatística com a ordem de magnitude verificada nos dados populacionais."
                )

            st.markdown(
                f"""
                * **Verdade Factual da População:** Na população completa (**{N_pop_txt}** registros), a proporção real de acidentes com vítimas fatais é de **{p_fatais_pop_txt}** ({k_fatais_pop:,} acidentes). Portanto, na realidade do censo, a proporção é determinística e factual **{rel_real}** a 2,00% por **{diff_real_txt} p.p.**
                * **Eficácia da Cobertura:** {'O intervalo de confiança de ' + conf_str + ' calculado a partir da amostra capturou o parâmetro populacional verdadeiro com sucesso.' if fatais_coberto else 'Houve desvio amostral em relação ao intervalo de ' + conf_str + ', o que é estatisticamente esperado em até ' + complemento_alfa + ' das amostras possíveis.'}
                * {conclusao_metodologica}
                """
            )
        else:
            N_pop_txt = f"{N_pop}".replace(",", ".")
            st.markdown(
                f"""
                * Na população completa (**{N_pop_txt}** registros), a proporção de acidentes fatais é de **{p_fatais_pop * 100:.2f}%** ({k_fatais_pop:,} acidentes).
                * Gere uma amostra na aba **'🧪 Amostra'** para confrontar as estimativas amostrais com os parâmetros reais da população.
                """
            )

    
    
    ####################################################################################################################################

    ## Página Inferência 2 do dashboard

    elif pagina_selecionada == "Inferência 2":

        # ---------------------------------------------------------
        # Recuperação dos parâmetros selecionados na página Amostra
        # ---------------------------------------------------------
        z_scores = {
            80: 1.282,
            85: 1.440,
            90: 1.645,
            95: 1.960,
            99: 2.576
        }

        grau_conf_val = int(st.session_state.get("confianca_memoria", 95))
        margem_erro_val = float(st.session_state.get("erro_memoria", 5.0))
        z_critico = z_scores.get(grau_conf_val, 1.960)

        # Formatações para exibição em textos e legendas
        conf_str = f"{grau_conf_val}%"
        z_crit_str = f"{z_critico:.2f}".replace(".", ",")
        erro_str = f"{margem_erro_val:.1f}%".replace(".", ",")

        st.markdown(rf"""
        
        ## Inferência estatística e análise dos resultados 🔮 
        
        ---
        
        ### Inferência 2: Acidentes com Feridos em Motos vs. Carros
        
        * **Pergunta investigativa:**
          A proporção de acidentes de trânsito com feridos que envolvem motocicletas é superior ou inferior à proporção dos acidentes com feridos que envolvem carros?

        * **Variáveis investigadas:**
          * $p_{{moto\_feridos}}$: proporção de acidentes com `moto > 0` e `feridos > 0` em relação ao total.
          * $p_{{carro\_feridos}}$: proporção de acidentes com `auto > 0` e `feridos > 0` em relação ao total.

        * **Hipóteses estatísticas:**
          * **Hipótese nula ($H_0$):** $p_{{moto\_feridos}} = p_{{carro\_feridos}}$
          * **Hipótese alternativa ($H_1$):** $p_{{moto\_feridos}} \neq p_{{carro\_feridos}}$

        * **Parâmetros Amostrais Adotados:** Grau de Confiança de **{conf_str}** ($Z_{{crítico}} = {z_crit_str}$) e Margem de Erro de **{erro_str}**.

        ---
        """)

        # ---------------------------------------------------------
        # 1. PARÂMETROS REAIS DA POPULAÇÃO (Censo / Ground Truth)
        # ---------------------------------------------------------
        N_pop = len(df_selected)
        k_moto_pop = ((df_selected["moto"] > 0) & (df_selected["feridos"] > 0)).sum()
        p_moto_pop = k_moto_pop / N_pop if N_pop > 0 else 0.0

        k_carro_pop = ((df_selected["auto"] > 0) & (df_selected["feridos"] > 0)).sum()
        p_carro_pop = k_carro_pop / N_pop if N_pop > 0 else 0.0

        diff_real_pp = (p_moto_pop - p_carro_pop) * 100

        # ---------------------------------------------------------
        # 2. TESTE DE INFERÊNCIA NA AMOSTRA
        # ---------------------------------------------------------
        st.markdown("### 🧪 1. Inferência Estatística a partir da Amostra")

        if "df_amostrado" in st.session_state:
            df_amostra = st.session_state["df_amostrado"]
            n_obs = len(df_amostra)

            # Fator de Correção de População Finita (FCPF)
            fcpf = np.sqrt((N_pop - n_obs) / (N_pop - 1)) if N_pop > n_obs else 1.0

            # Cálculos amostrais
            k_moto_amostra = ((df_amostra["moto"] > 0) & (df_amostra["feridos"] > 0)).sum()
            p_moto_hat = k_moto_amostra / n_obs

            k_carro_amostra = ((df_amostra["auto"] > 0) & (df_amostra["feridos"] > 0)).sum()
            p_carro_hat = k_carro_amostra / n_obs

            # Intervalos de Confiança calculados dinamicamente com o Z_crítico atual
            se_moto = np.sqrt((p_moto_hat * (1 - p_moto_hat)) / n_obs) * fcpf
            ic_moto_inf = max(0.0, p_moto_hat - (z_critico * se_moto))
            ic_moto_sup = min(1.0, p_moto_hat + (z_critico * se_moto))

            se_carro = np.sqrt((p_carro_hat * (1 - p_carro_hat)) / n_obs) * fcpf
            ic_carro_inf = max(0.0, p_carro_hat - (z_critico * se_carro))
            ic_carro_sup = min(1.0, p_carro_hat + (z_critico * se_carro))

            # Teste Z bicaudal para diferença entre duas proporções
            p_pool = (k_moto_amostra + k_carro_amostra) / (2 * n_obs)
            se_diff = np.sqrt(p_pool * (1 - p_pool) * (2 / n_obs)) * fcpf
            z_stat = (p_moto_hat - p_carro_hat) / se_diff if se_diff > 0 else 0.0
            rejeita_h0 = abs(z_stat) > z_critico

            col_a1, col_a2 = st.columns(2)
            
            p_moto_str = f"{p_moto_hat * 100:.2f}%".replace(".", ",")
            ic_moto_str = f"[{ic_moto_inf * 100:.2f}%, {ic_moto_sup * 100:.2f}%]".replace(".", ",")
            contagem_moto_str = f"({k_moto_amostra:,}/{n_obs:,})".replace(",", ".")

            p_carro_str = f"{p_carro_hat * 100:.2f}%".replace(".", ",")
            ic_carro_str = f"[{ic_carro_inf * 100:.2f}%, {ic_carro_sup * 100:.2f}%]".replace(".", ",")
            contagem_carro_str = f"({k_carro_amostra:,}/{n_obs:,})".replace(",", ".")

            with col_a1:
                st.info(rf"**Estimativa: Moto com Feridos ($\hat{{p}}_{{moto}}$)**")
                st.markdown(
                    rf"""
                    <div style="font-size: 1.05rem; line-height: 1.8;">
                        • <b>Proporção na amostra:</b> 
                        <span style="font-size: 1.45rem; font-weight: 700; color: #1f77b4; margin-left: 4px;">
                            {p_moto_str}
                        </span> 
                        <span style="font-size: 0.95rem; color: #555555;">{contagem_moto_str}</span>
                        <br>
                        • <b>Intervalo de Confiança ({conf_str}):</b> 
                        <span style="font-size: 1.30rem; font-weight: 600; color: #2c3e50; margin-left: 4px;">
                            {ic_moto_str}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_a2:
                st.info(rf"**Estimativa: Carro com Feridos ($\hat{{p}}_{{carro}}$)**")
                st.markdown(
                    rf"""
                    <div style="font-size: 1.05rem; line-height: 1.8;">
                        • <b>Proporção na amostra:</b> 
                        <span style="font-size: 1.45rem; font-weight: 700; color: #1f77b4; margin-left: 4px;">
                            {p_carro_str}
                        </span> 
                        <span style="font-size: 0.95rem; color: #555555;">{contagem_carro_str}</span>
                        <br>
                        • <b>Intervalo de Confiança ({conf_str}):</b> 
                        <span style="font-size: 1.30rem; font-weight: 600; color: #2c3e50; margin-left: 4px;">
                            {ic_carro_str}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            z_stat_str = f"{abs(z_stat):.2f}".replace(".", ",")

            if rejeita_h0:
                rel_amostra = "inferior" if p_moto_hat < p_carro_hat else "superior"
                st.success(
                    rf"**Conclusão da Inferência Amostral: Rejeita-se $H_0$!** "
                    rf"Com base na amostra coletada ($n = {n_obs:,}$), a proporção de acidentes com feridos envolvendo motos (`{p_moto_str}`) "
                    rf"é **estatisticamente {rel_amostra}** à de carros (`{p_carro_str}`) ao nível de {conf_str} de confiança (|Z| = {z_stat_str} > {z_crit_str}).".replace(",", ".")
                )
            else:
                st.warning(
                    rf"**Conclusão da Inferência Amostral: Não se rejeita $H_0$.** "
                    rf"Com base na amostra ($n = {n_obs:,}$), a diferença observada não possui significância estatística ao nível de {conf_str} (|Z| = {z_stat_str} <= {z_crit_str}).".replace(",", ".")
                )
        else:
            st.info("💡 *Nenhuma amostra foi gerada até o momento. Acesse a página '🧪 Amostra' e gere uma amostra aleatória para visualizar o teste inferencial.*")

        # ---------------------------------------------------------
        # 3. PARÂMETROS REAIS DA POPULAÇÃO (Ground Truth)
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🌐 2. Parâmetros Reais da População (Censo / Ground Truth)")
        st.caption("Como estes dados compreendem a totalidade dos registros da população selecionada, os valores abaixo representam os parâmetros reais exatos ($p$), dispensando intervalos de confiança ou testes de hipótese probabilísticos.")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric(
                label="Proporção Real em Motos (p_moto)",
                value=f"{p_moto_pop * 100:.2f}%".replace(".", ","),
                help=f"{k_moto_pop:,} acidentes de {N_pop:,}".replace(",", ".")
            )
        with col_p2:
            st.metric(
                label="Proporção Real em Carros (p_carro)",
                value=f"{p_carro_pop * 100:.2f}%".replace(".", ","),
                help=f"{k_carro_pop:,} acidentes de {N_pop:,}".replace(",", ".")
            )
        with col_p3:
            st.metric(
                label="Diferença Real Populacional (p_moto - p_carro)",
                value=f"{diff_real_pp:.2f}".replace(".", ",") + " p.p.",
                help="Diferença determinística observada na população."
            )

        # ---------------------------------------------------------
        # 4. VALIDAÇÃO E INTERPRETAÇÃO DA INFERÊNCIA
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### ⚖️ 3. Avaliação e Validação da Inferência Amostral")

        if "df_amostrado" in st.session_state:
            moto_coberto = ic_moto_inf <= p_moto_pop <= ic_moto_sup
            carro_coberto = ic_carro_inf <= p_carro_pop <= ic_carro_sup

            df_grafico_val = pd.DataFrame([
                {
                    "categoria": "Moto com Feridos",
                    "p_hat": p_moto_hat * 100,
                    "ic_inf": ic_moto_inf * 100,
                    "ic_sup": ic_moto_sup * 100,
                    "p_real": p_moto_pop * 100,
                    "rotulo_amostra": f"{p_moto_hat * 100:.2f}% (IC {conf_str}: [{ic_moto_inf * 100:.2f}% - {ic_moto_sup * 100:.2f}%])".replace(".", ",")
                },
                {
                    "categoria": "Carro com Feridos",
                    "p_hat": p_carro_hat * 100,
                    "ic_inf": ic_carro_inf * 100,
                    "ic_sup": ic_carro_sup * 100,
                    "p_real": p_carro_pop * 100,
                    "rotulo_amostra": f"{p_carro_hat * 100:.2f}% (IC {conf_str}: [{ic_carro_inf * 100:.2f}% - {ic_carro_sup * 100:.2f}%])".replace(".", ",")
                }
            ])

            barras_amostra = (
                alt.Chart(df_grafico_val)
                .mark_rule(strokeWidth=4, color="#de2d26")
                .encode(
                    x=alt.X(
                        "ic_inf:Q",
                        title="Proporção (%)",
                        scale=alt.Scale(
                            domain=[
                                max(0, min(df_grafico_val["ic_inf"].min(), df_grafico_val["p_real"].min()) * 0.85),
                                max(df_grafico_val["ic_sup"].max(), df_grafico_val["p_real"].max()) * 1.25
                            ]
                        )
                    ),
                    x2="ic_sup:Q",
                    y=alt.Y("categoria:N", title="Categoria", sort=["Moto com Feridos", "Carro com Feridos"])
                )
            )

            pontos_amostra = (
                alt.Chart(df_grafico_val)
                .mark_point(size=130, filled=True, color="#de2d26")
                .encode(
                    x="p_hat:Q",
                    y=alt.Y("categoria:N", sort=["Moto com Feridos", "Carro com Feridos"]),
                    tooltip=[
                        alt.Tooltip("categoria:N", title="Categoria"),
                        alt.Tooltip("p_hat:Q", title="Estimativa Amostral (%)", format=".2f"),
                        alt.Tooltip("ic_inf:Q", title=f"IC Inf ({conf_str})", format=".2f"),
                        alt.Tooltip("ic_sup:Q", title=f"IC Sup ({conf_str})", format=".2f")
                    ]
                )
            )

            pontos_real = (
                alt.Chart(df_grafico_val)
                .mark_tick(size=25, thickness=4, color="#1b7837")
                .encode(
                    x="p_real:Q",
                    y=alt.Y("categoria:N", sort=["Moto com Feridos", "Carro com Feridos"]),
                    tooltip=[
                        alt.Tooltip("categoria:N", title="Categoria"),
                        alt.Tooltip("p_real:Q", title="Parâmetro Real População (%)", format=".2f")
                    ]
                )
            )

            texto_amostra = (
                alt.Chart(df_grafico_val)
                .mark_text(align="left", baseline="middle", dx=12, color="#333333")
                .encode(
                    x="ic_sup:Q",
                    y=alt.Y("categoria:N", sort=["Moto com Feridos", "Carro com Feridos"]),
                    text="rotulo_amostra:N"
                )
            )

            grafico_validacao = (
                (barras_amostra + pontos_amostra + pontos_real + texto_amostra)
                .properties(
                    height=200,
                    title=f"Validação: Estimativa Pontual e IC {conf_str} da Amostra (Vermelho) vs. Parâmetro Real Populacional (Traço Verde)"
                )
            )

            st.altair_chart(grafico_validacao, width="stretch")

            # Tabela de validação com cabeçalho dinâmico
            df_tabela_validacao = pd.DataFrame([
                {
                    "Variável": "Moto com Feridos",
                    "Amostra (p_hat)": f"{p_moto_hat * 100:.2f}%".replace(".", ","),
                    f"IC {conf_str} da Amostra": f"[{ic_moto_inf * 100:.2f}%, {ic_moto_sup * 100:.2f}%]".replace(".", ","),
                    "Parâmetro Real População (p)": f"{p_moto_pop * 100:.2f}%".replace(".", ","),
                    "O Parâmetro Real está no IC?": "✅ Sim" if moto_coberto else "❌ Não"
                },
                {
                    "Variável": "Carro com Feridos",
                    "Amostra (p_hat)": f"{p_carro_hat * 100:.2f}%".replace(".", ","),
                    f"IC {conf_str} da Amostra": f"[{ic_carro_inf * 100:.2f}%, {ic_carro_sup * 100:.2f}%]".replace(".", ","),
                    "Parâmetro Real População (p)": f"{p_carro_pop * 100:.2f}%".replace(".", ","),
                    "O Parâmetro Real está no IC?": "✅ Sim" if carro_coberto else "❌ Não"
                }
            ])

            st.dataframe(df_tabela_validacao, hide_index=True, width="stretch")

            # Conclusão metodológica dinâmica
            if rejeita_h0:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica:** A amostragem aleatória com nível de confiança de {conf_str} demonstrou alta eficácia, "
                    f"permitindo identificar a diferença real existente na população e chegar à mesma decisão prática (|Z| = {z_stat_str} > {z_crit_str}) "
                    f"sem a necessidade de processar a totalidade dos dados."
                )
            else:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica (Alerta de Poder Amostral):** A amostra não rejeitou a hipótese nula $H_0$ (|Z| = {z_stat_str} <= {z_crit_str}), "
                    f"divergindo da realidade observada na população (onde há diferença real). Esse comportamento "
                    f"ilustra o risco do **Erro Tipo II (falso negativo)** decorrente da variabilidade amostral ou "
                    f"de uma margem de erro ({erro_str}) ampla para o tamanho do efeito real observado."
                )

            N_pop_txt = f"{N_pop}".replace(",", ".")
            p_moto_pop_txt = f"{p_moto_pop * 100:.2f}%".replace(".", ",")
            p_carro_pop_txt = f"{p_carro_pop * 100:.2f}%".replace(".", ",")
            diff_real_txt = f"{abs(diff_real_pp):.2f}".replace(".", ",")
            rel_real = "inferior" if p_moto_pop < p_carro_pop else "superior"
            complemento_alfa = f"{100 - grau_conf_val}%"

            st.markdown(
                f"""
                * **Verdade Factual da População:** Na população completa (**{N_pop_txt}** registros), a proporção real de acidentes com feridos envolvendo motocicletas é de **{p_moto_pop_txt}**, enquanto a de automóveis é de **{p_carro_pop_txt}**. Portanto, na realidade dos fatos, a proporção em motos é determinística e factual **{rel_real}** à de carros por **{diff_real_txt} p.p.**
                * **Eficácia da Cobertura:** {'Ambos os intervalos de confiança de ' + conf_str + ' calculados a partir da amostra capturaram os parâmetros populacionais verdadeiros com sucesso.' if moto_coberto and carro_coberto else 'Houve desvio amostral em pelo menos uma das variáveis em relação ao intervalo de ' + conf_str + ', o que é estatisticamente esperado em até ' + complemento_alfa + ' das amostras possíveis.'}
                * {conclusao_metodologica}
                """
            )
        else:
            N_pop_txt = f"{N_pop}".replace(",", ".")
            p_moto_pop_txt = f"{p_moto_pop * 100:.2f}%".replace(".", ",")
            p_carro_pop_txt = f"{p_carro_pop * 100:.2f}%".replace(".", ",")
            st.markdown(
                f"""
                * Na população completa (**{N_pop_txt}** registros), a proporção de acidentes com feridos envolvendo motos é de **{p_moto_pop_txt}** contra **{p_carro_pop_txt}** nos acidentes envolvendo carros.
                * Gere uma amostra na aba **'🧪 Amostra'** para confrontar as estimativas amostrais com os parâmetros reais da população.
                """
            )



    ####################################################################################################################################

    ## Página Inferência 3 do dashboard

    elif pagina_selecionada == "Inferência 3":

        # ---------------------------------------------------------
        # Recuperação dos parâmetros selecionados na página Amostra
        # ---------------------------------------------------------
        z_scores = {
            80: 1.282,
            85: 1.440,
            90: 1.645,
            95: 1.960,
            99: 2.576
        }

        grau_conf_val = int(st.session_state.get("confianca_memoria", 95))
        margem_erro_val = float(st.session_state.get("erro_memoria", 5.0))
        z_critico = z_scores.get(grau_conf_val, 1.960)

        # Formatações para exibição
        conf_str = f"{grau_conf_val}%"
        z_crit_str = f"{z_critico:.2f}".replace(".", ",")
        erro_str = f"{margem_erro_val:.1f}%".replace(".", ",")
        mu_ref = 1.73  # Média de referência das capitais brasileiras
        mu_ref_str = "1,730"
        
        st.markdown(rf"""
        
        ## Inferência estatística e análise dos resultados 🔮 
        
        ---
        
        ### Inferência 3: Média de Feridos em Porto Alegre vs. Capitais Brasileiras
        
        * **Pergunta investigativa:**
          A média de feridos por acidente ocorridos em Porto Alegre é diferente de 1,73, que é a média aproximada para as capitais brasileiras?

        * **Variável investigada:**
          * $\mu_{{feridos}}$: média do número de feridos nos acidentes de trânsito em Porto Alegre.

        * **Hipóteses estatísticas:**
          * **Hipótese nula ($H_0$):** $\mu_{{feridos}} = 1,73$ ($\mu_0 = 1,73$)
          * **Hipótese alternativa ($H_1$):** $\mu_{{feridos}} \neq 1,73$ ($\mu_0 \neq 1,73$)

        * **Parâmetros Amostrais Adotados:** Grau de Confiança de **{conf_str}** ($Z_{{crítico}} = {z_crit_str}$) e Margem de Erro de **{erro_str}**.

        ---
        """)

        # ---------------------------------------------------------
        # 1. PARÂMETROS REAIS DA POPULAÇÃO (Censo / Ground Truth)
        # ---------------------------------------------------------
        N_pop = len(df_selected)
        total_feridos_pop = df_selected["feridos"].sum()
        mu_pop = df_selected["feridos"].mean() if N_pop > 0 else 0.0
        sigma_pop = df_selected["feridos"].std(ddof=0) if N_pop > 0 else 0.0
        diff_pop_ref = mu_pop - mu_ref

        # ---------------------------------------------------------
        # 2. TESTE DE INFERÊNCIA NA AMOSTRA
        # ---------------------------------------------------------
        st.markdown("### 🧪 1. Inferência Estatística a partir da Amostra")

        if "df_amostrado" in st.session_state:
            df_amostra = st.session_state["df_amostrado"]
            n_obs = len(df_amostra)

            if n_obs >= 2:
                # Fator de Correção de População Finita (FCPF)
                fcpf = np.sqrt((N_pop - n_obs) / (N_pop - 1)) if N_pop > n_obs else 1.0

                # Estimadores Amostrais
                xbar_amostra = df_amostra["feridos"].mean()
                s_amostra = df_amostra["feridos"].std(ddof=1)
                total_feridos_amostra = df_amostra["feridos"].sum()

                # Erro Padrão da Média
                se_amostra = (s_amostra / np.sqrt(n_obs)) * fcpf

                # Intervalo de Confiança para a Média
                ic_inf = max(0.0, xbar_amostra - (z_critico * se_amostra))
                ic_sup = xbar_amostra + (z_critico * se_amostra)

                # Teste Z para uma média sob H0
                z_stat = (xbar_amostra - mu_ref) / se_amostra if se_amostra > 0 else 0.0
                rejeita_h0 = abs(z_stat) > z_critico

                # Strings formatadas
                xbar_str = f"{xbar_amostra:.3f}".replace(".", ",")
                s_str = f"{s_amostra:.3f}".replace(".", ",")
                ic_str = f"[{ic_inf:.3f}, {ic_sup:.3f}]".replace(".", ",")
                contagem_str = f"({total_feridos_amostra:,.0f} feridos em {n_obs:,} acid.)".replace(",", ".")
                z_stat_str = f"{abs(z_stat):.2f}".replace(".", ",")

                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.info(rf"**Estimativa Amostral: Média de Feridos ($\bar{{x}}_{{feridos}}$)**")
                    st.markdown(
                        rf"""
                        <div style="font-size: 1.05rem; line-height: 1.8;">
                            • <b>Média na amostra:</b> 
                            <span style="font-size: 1.45rem; font-weight: 700; color: #1f77b4; margin-left: 4px;">
                                {xbar_str}
                            </span> 
                            <span style="font-size: 0.95rem; color: #555555;">{contagem_str}</span>
                            <br>
                            • <b>Intervalo de Confiança ({conf_str}):</b> 
                            <span style="font-size: 1.30rem; font-weight: 600; color: #2c3e50; margin-left: 4px;">
                                {ic_str}
                            </span>
                            <span style="font-size: 0.90rem; color: #555555;">(Desvio <i>s</i> = {s_str})</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_a2:
                    st.info(r"**Hipótese de Teste de Referência ($H_0$)**")
                    st.markdown(
                        rf"""
                        <div style="font-size: 1.05rem; line-height: 1.8;">
                            • <b>Média de Referência (&mu;<sub>0</sub>):</b> 
                            <span style="font-size: 1.45rem; font-weight: 700; color: #7fcdbb; margin-left: 4px;">
                                {mu_ref_str}
                            </span>
                            <span style="font-size: 0.95rem; color: #555555;">feridos/acid.</span>
                            <br>
                            • <b>Estatística de Teste (|<i>Z</i>|):</b> 
                            <span style="font-size: 1.30rem; font-weight: 600; color: #2c3e50; margin-left: 4px;">
                                {z_stat_str}
                            </span> 
                            <span style="font-size: 0.95rem; color: #555555;">(Crítico: {z_crit_str})</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if rejeita_h0:
                    rel_amostra = "superior" if xbar_amostra > mu_ref else "inferior"
                    st.success(
                        rf"**Conclusão da Inferência Amostral: Rejeita-se $H_0$!** "
                        rf"Com base na amostra coletada ($n = {n_obs:,}$ registros), a média de feridos por acidente em Porto Alegre (`{xbar_str}`) "
                        rf"é **estatisticamente {rel_amostra}** ao parâmetro de referência das capitais (1,730 feridos/acid.) ao nível de {conf_str} de confiança (|Z| = {z_stat_str} > {z_crit_str}).".replace(",", ".")
                    )
                else:
                    st.warning(
                        rf"**Conclusão da Inferência Amostral: Não se rejeita $H_0$.** "
                        rf"Com base na amostra ($n = {n_obs:,}$ registros), não há evidência estatística suficiente ao nível de {conf_str} "
                        rf"para afirmar que a média de feridos em Porto Alegre (`{xbar_str}`) difira da média das capitais brasileiras (1,730) (|Z| = {z_stat_str} <= {z_crit_str}).".replace(",", ".")
                    )
            else:
                st.warning("A amostra atual não contém observações suficientes para realizar o teste de média.")
        else:
            st.info("💡 *Nenhuma amostra foi gerada até o momento. Acesse a página '🧪 Amostra' e gere uma amostra aleatória para visualizar o teste inferencial.*")

        # ---------------------------------------------------------
        # 3. PARÂMETROS REAIS DA POPULAÇÃO (Ground Truth)
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🌐 2. Parâmetros Reais da População (Censo / Ground Truth)")
        st.caption(r"Como estes dados compreendem a totalidade dos registros da população selecionada, os valores abaixo representam os parâmetros populacionais exatos ($\mu$ e $\sigma$), dispensando intervalos de confiança ou testes de hipótese probabilísticos.")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric(
                label="Média Real de Porto Alegre (μ_feridos)",
                value=f"{mu_pop:.3f}".replace(".", ","),
                help=f"{total_feridos_pop:,.0f} feridos em {N_pop:,} acidentes (Desvio-Padrão σ = {sigma_pop:.2f})".replace(",", ".")
            )
        with col_p2:
            st.metric(
                label="Média de Referência Capitais (μ_0)",
                value=mu_ref_str,
                help="Parâmetro aproximado para as grandes capitais brasileiras (1,73 feridos/acidente)."
            )
        with col_p3:
            st.metric(
                label="Diferença Real Populacional (μ - 1,73)",
                value=f"{diff_pop_ref:.3f}".replace(".", ",") + " feridos/acid.",
                help="Diferença determinística observada na população."
            )

        # ---------------------------------------------------------
        # 4. VALIDAÇÃO E INTERPRETAÇÃO DA INFERÊNCIA
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### ⚖️ 3. Avaliação e Validação da Inferência Amostral")

        if "df_amostrado" in st.session_state and n_obs >= 2:
            mu_coberto = ic_inf <= mu_pop <= ic_sup
            ref_no_ic = ic_inf <= mu_ref <= ic_sup

            df_grafico_val = pd.DataFrame([
                {
                    "categoria": "Média de Feridos",
                    "xbar": xbar_amostra,
                    "ic_inf": ic_inf,
                    "ic_sup": ic_sup,
                    "mu_real": mu_pop,
                    "mu_ref": mu_ref,
                    "rotulo_amostra": f"{xbar_amostra:.3f} (IC {conf_str}: [{ic_inf:.3f} - {ic_sup:.3f}])".replace(".", ",")
                }
            ])

            # Barra do IC da Amostra
            barras_amostra = (
                alt.Chart(df_grafico_val)
                .mark_rule(strokeWidth=4, color="#de2d26")
                .encode(
                    x=alt.X(
                        "ic_inf:Q",
                        title="Média de Feridos por Acidente",
                        scale=alt.Scale(
                            domain=[
                                max(0, min(df_grafico_val["ic_inf"].min(), df_grafico_val["mu_real"].min(), mu_ref) * 0.85),
                                max(df_grafico_val["ic_sup"].max(), df_grafico_val["mu_real"].max(), mu_ref) * 1.25
                            ]
                        )
                    ),
                    x2="ic_sup:Q",
                    y=alt.Y("categoria:N", title=None)
                )
            )

            # Ponto da estimativa pontual da amostra (\bar{x})
            pontos_amostra = (
                alt.Chart(df_grafico_val)
                .mark_point(size=130, filled=True, color="#de2d26")
                .encode(
                    x="xbar:Q",
                    y=alt.Y("categoria:N"),
                    tooltip=[
                        alt.Tooltip("categoria:N", title="Indicador"),
                        alt.Tooltip("xbar:Q", title="Média Amostral", format=".3f"),
                        alt.Tooltip("ic_inf:Q", title=f"IC Inf ({conf_str})", format=".3f"),
                        alt.Tooltip("ic_sup:Q", title=f"IC Sup ({conf_str})", format=".3f")
                    ]
                )
            )

            # Traço do Parâmetro Real da População (Ground Truth)
            pontos_real = (
                alt.Chart(df_grafico_val)
                .mark_tick(size=25, thickness=4, color="#1b7837")
                .encode(
                    x="mu_real:Q",
                    y=alt.Y("categoria:N"),
                    tooltip=[
                        alt.Tooltip("categoria:N", title="Indicador"),
                        alt.Tooltip("mu_real:Q", title="Média Real População (μ)", format=".3f")
                    ]
                )
            )

            # Linha vertical pontilhada indicando o valor de referência H0 (1,73)
            df_ref_linha = pd.DataFrame([{"ref": mu_ref}])
            linha_ref = (
                alt.Chart(df_ref_linha)
                .mark_rule(strokeDash=[6, 4], strokeWidth=2, color="#001F54")
                .encode(
                    x="ref:Q",
                    tooltip=[alt.Tooltip("ref:Q", title="Valor de Referência H0 (μ0)", format=".3f")]
                )
            )

            # Rótulo em texto com a média amostral e o IC
            texto_amostra = (
                alt.Chart(df_grafico_val)
                .mark_text(align="left", baseline="middle", dx=12, color="#333333")
                .encode(
                    x="ic_sup:Q",
                    y=alt.Y("categoria:N"),
                    text="rotulo_amostra:N"
                )
            )

            grafico_validacao = (
                (barras_amostra + pontos_amostra + pontos_real + linha_ref + texto_amostra)
                .properties(
                    height=160,
                    title=f"Validação: IC {conf_str} da Amostra (Vermelho) vs. Média Real (Verde) vs. Referência H0 de 1,73 (Linha Azul)"
                )
            )

            st.altair_chart(grafico_validacao, width="stretch")

            # Tabela de validação
            df_tabela_validacao = pd.DataFrame([
                {
                    "Indicador": "Média de Feridos",
                    "Amostra (x̄)": f"{xbar_amostra:.3f}".replace(".", ","),
                    f"IC {conf_str} da Amostra": f"[{ic_inf:.3f}, {ic_sup:.3f}]".replace(".", ","),
                    "Média Real População (μ)": f"{mu_pop:.3f}".replace(".", ","),
                    "O Parâmetro Real está no IC?": "✅ Sim" if mu_coberto else "❌ Não",
                    "Valor 1,73 está no IC?": "Sim (H0 Aceita)" if ref_no_ic else "Não (H0 Rejeitada)"
                }
            ])

            st.dataframe(df_tabela_validacao, hide_index=True, width="stretch")

            # Conclusão metodológica dinâmica
            N_pop_txt = f"{N_pop}".replace(",", ".")
            mu_pop_txt = f"{mu_pop:.3f}".replace(".", ",")
            diff_real_txt = f"{abs(diff_pop_ref):.3f}".replace(".", ",")
            sigma_pop_txt = f"{sigma_pop:.2f}".replace(".", ",")
            rel_real = "superior" if mu_pop > mu_ref else "inferior"
            complemento_alfa = f"{100 - grau_conf_val}%"

            # Avalia se a decisão da amostra corresponde ao que de fato ocorre na população
            pop_diverge_ref = abs(diff_pop_ref) >= 0.05

            if rejeita_h0 and pop_diverge_ref:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica:** A amostragem aleatória com nível de confiança de {conf_str} demonstrou alta eficácia, "
                    f"permitindo detectar que a média de feridos em Porto Alegre diverge significativamente do parâmetro de 1,730 (|Z| = {z_stat_str} > {z_crit_str}) "
                    f"sem a necessidade de analisar a totalidade dos dados."
                )
            elif not rejeita_h0 and pop_diverge_ref:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica (Alerta de Poder Amostral):** A amostra não rejeitou a hipótese nula $H_0$ (|Z| = {z_stat_str} <= {z_crit_str}), "
                    f"enquanto na população completa a média real de acidentes é de fato {rel_real} a 1,730 por {diff_real_txt} feridos/acidente. "
                    f"Esse comportamento ilustra a ocorrência do **Erro Tipo II (falso negativo)** decorrente da variabilidade amostral ou "
                    f"de um tamanho de amostra insuficiente para rejeitar a hipótese sob a margem de erro ({erro_str}) estipulada."
                )
            else:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica:** A inferência amostral indicou proximidade estatística entre a média observada e o limiar de 1,730 feridos/acidente (|Z| = {z_stat_str} <= {z_crit_str}), "
                    f"mostrando consistência metodológica com a ordem de magnitude verificada nos registros populacionais."
                )

            st.markdown(
                rf"""
                * **Verdade Factual da População:** Na população completa (**{N_pop_txt}** registros válidos), a média real de feridos por acidente de trânsito em Porto Alegre é de **{mu_pop_txt}** (desvio $\sigma$ = **{sigma_pop_txt}**). Portanto, na realidade do censo, a severidade média de feridos na capital gaúcha é factual e determinística **{rel_real}** à média das capitais brasileiras por **{diff_real_txt} feridos/acidente**.
                * **Eficácia da Cobertura:** {'O intervalo de confiança de ' + conf_str + ' calculado a partir da amostra capturou o parâmetro populacional verdadeiro com sucesso.' if mu_coberto else 'Houve desvio amostral em relação ao intervalo de ' + conf_str + ', o que é estatisticamente esperado em até ' + complemento_alfa + ' das amostras possíveis.'}
                * {conclusao_metodologica}
                """
            )
        else:
            N_pop_txt = f"{N_pop}".replace(",", ".")
            mu_pop_txt = f"{mu_pop:.3f}".replace(".", ",")
            st.markdown(
                f"""
                * Na população completa (**{N_pop_txt}** registros), a média real de feridos por acidente em Porto Alegre é de **{mu_pop_txt}** (contra 1,730 de referência).
                * Gere uma amostra na aba **'🧪 Amostra'** para confrontar as estimativas amostrais com os parâmetros reais da população.
                """
            )



    ####################################################################################################################################

    ## Página Inferência 4 do dashboard

    elif pagina_selecionada == "Inferência 4":

        # ---------------------------------------------------------
        # Recuperação dos parâmetros selecionados na página Amostra
        # ---------------------------------------------------------
        z_scores = {
            80: 1.282,
            85: 1.440,
            90: 1.645,
            95: 1.960,
            99: 2.576
        }

        grau_conf_val = int(st.session_state.get("confianca_memoria", 95))
        margem_erro_val = float(st.session_state.get("erro_memoria", 5.0))
        z_critico = z_scores.get(grau_conf_val, 1.960)

        # Formatações para exibição em textos e legendas
        conf_str = f"{grau_conf_val}%"
        z_crit_str = f"{z_critico:.2f}".replace(".", ",")
        erro_str = f"{margem_erro_val:.1f}%".replace(".", ",")

        st.markdown(rf"""
        
        ## Inferência estatística e análise dos resultados 🔮 
        
        ---
        
        ### Inferência 4: Média de Feridos em Acidentes (Dia vs. Noite)
        
        * **Pergunta investigativa:**
          A média de feridos por acidente nos ocorridos durante o dia é igual à média de feridos por acidente nos ocorridos durante a noite?

        * **Variáveis investigadas:**
          * $\mu_{{dia\_feridos}}$: média do número de feridos em acidentes ocorridos durante o dia (`noite_dia == 'DIA'`).
          * $\mu_{{noite\_feridos}}$: média do número de feridos em acidentes ocorridos durante a noite (`noite_dia == 'NOITE'`).

        * **Hipóteses estatísticas:**
          * **Hipótese nula ($H_0$):** $\mu_{{dia\_feridos}} = \mu_{{noite\_feridos}}$
          * **Hipótese alternativa ($H_1$):** $\mu_{{dia\_feridos}} \neq \mu_{{noite\_feridos}}$

        * **Parâmetros Amostrais Adotados:** Grau de Confiança de **{conf_str}** ($Z_{{crítico}} = {z_crit_str}$) e Margem de Erro de **{erro_str}**.

        ---
        """)

        # ---------------------------------------------------------
        # 1. PARÂMETROS REAIS DA POPULAÇÃO (Censo / Ground Truth)
        # ---------------------------------------------------------
        df_pop_dia = df_selected[df_selected["noite_dia"] == "DIA"]
        df_pop_noite = df_selected[df_selected["noite_dia"] == "NOITE"]

        N_pop_dia = len(df_pop_dia)
        N_pop_noite = len(df_pop_noite)
        N_pop_total = len(df_selected)

        mu_dia_pop = df_pop_dia["feridos"].mean() if N_pop_dia > 0 else 0.0
        mu_noite_pop = df_pop_noite["feridos"].mean() if N_pop_noite > 0 else 0.0
        
        sigma_dia_pop = df_pop_dia["feridos"].std(ddof=0) if N_pop_dia > 0 else 0.0
        sigma_noite_pop = df_pop_noite["feridos"].std(ddof=0) if N_pop_noite > 0 else 0.0

        diff_real_mu = mu_dia_pop - mu_noite_pop

        # ---------------------------------------------------------
        # 2. TESTE DE INFERÊNCIA NA AMOSTRA
        # ---------------------------------------------------------
        st.markdown("### 🧪 1. Inferência Estatística a partir da Amostra")

        if "df_amostrado" in st.session_state:
            df_amostra = st.session_state["df_amostrado"]
            df_amostra_dia = df_amostra[df_amostra["noite_dia"] == "DIA"]
            df_amostra_noite = df_amostra[df_amostra["noite_dia"] == "NOITE"]

            n_dia = len(df_amostra_dia)
            n_noite = len(df_amostra_noite)
            n_total = len(df_amostra)

            if n_dia >= 2 and n_noite >= 2:
                # Médias e Desvios Amostrais
                xbar_dia = df_amostra_dia["feridos"].mean()
                s_dia = df_amostra_dia["feridos"].std(ddof=1)

                xbar_noite = df_amostra_noite["feridos"].mean()
                s_noite = df_amostra_noite["feridos"].std(ddof=1)

                # Fator de Correção de População Finita (FCPF) por estrato/turno
                fcpf_dia = np.sqrt((N_pop_dia - n_dia) / (N_pop_dia - 1)) if N_pop_dia > n_dia else 1.0
                fcpf_noite = np.sqrt((N_pop_noite - n_noite) / (N_pop_noite - 1)) if N_pop_noite > n_noite else 1.0

                # Erros padrão das médias
                se_dia = (s_dia / np.sqrt(n_dia)) * fcpf_dia
                se_noite = (s_noite / np.sqrt(n_noite)) * fcpf_noite

                # Intervalos de Confiança
                ic_dia_inf = max(0.0, xbar_dia - (z_critico * se_dia))
                ic_dia_sup = xbar_dia + (z_critico * se_dia)

                ic_noite_inf = max(0.0, xbar_noite - (z_critico * se_noite))
                ic_noite_sup = xbar_noite + (z_critico * se_noite)

                # Erro padrão da diferença entre duas médias independentes
                se_diff = np.sqrt((se_dia ** 2) + (se_noite ** 2))
                z_stat = (xbar_dia - xbar_noite) / se_diff if se_diff > 0 else 0.0
                rejeita_h0 = abs(z_stat) > z_critico

                # Strings formatadas
                xbar_dia_str = f"{xbar_dia:.3f}".replace(".", ",")
                ic_dia_str = f"[{ic_dia_inf:.3f}, {ic_dia_sup:.3f}]".replace(".", ",")
                detalhes_dia_str = f"({df_amostra_dia['feridos'].sum():,.0f} feridos em {n_dia:,} acid.)".replace(",", ".")

                xbar_noite_str = f"{xbar_noite:.3f}".replace(".", ",")
                ic_noite_str = f"[{ic_noite_inf:.3f}, {ic_noite_sup:.3f}]".replace(".", ",")
                detalhes_noite_str = f"({df_amostra_noite['feridos'].sum():,.0f} feridos em {n_noite:,} acid.)".replace(",", ".")

                z_stat_str = f"{abs(z_stat):.2f}".replace(".", ",")

                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.info(rf"**Estimativa: Média de Feridos - DIA ($\bar{{x}}_{{dia}}$)**")
                    st.markdown(
                        rf"""
                        <div style="font-size: 1.05rem; line-height: 1.8;">
                            • <b>Média na amostra:</b> 
                            <span style="font-size: 1.45rem; font-weight: 700; color: #1f77b4; margin-left: 4px;">
                                {xbar_dia_str}
                            </span> 
                            <span style="font-size: 0.95rem; color: #555555;">{detalhes_dia_str}</span>
                            <br>
                            • <b>Intervalo de Confiança ({conf_str}):</b> 
                            <span style="font-size: 1.30rem; font-weight: 600; color: #2c3e50; margin-left: 4px;">
                                {ic_dia_str}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_a2:
                    st.info(rf"**Estimativa: Média de Feridos - NOITE ($\bar{{x}}_{{noite}}$)**")
                    st.markdown(
                        rf"""
                        <div style="font-size: 1.05rem; line-height: 1.8;">
                            • <b>Média na amostra:</b> 
                            <span style="font-size: 1.45rem; font-weight: 700; color: #1f77b4; margin-left: 4px;">
                                {xbar_noite_str}
                            </span> 
                            <span style="font-size: 0.95rem; color: #555555;">{detalhes_noite_str}</span>
                            <br>
                            • <b>Intervalo de Confiança ({conf_str}):</b> 
                            <span style="font-size: 1.30rem; font-weight: 600; color: #2c3e50; margin-left: 4px;">
                                {ic_noite_str}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if rejeita_h0:
                    rel_amostra = "superior" if xbar_dia > xbar_noite else "inferior"
                    st.success(
                        rf"**Conclusão da Inferência Amostral: Rejeita-se $H_0$!** "
                        rf"Com base na amostra coletada ($n = {n_total:,}$ registros válidos), a média de feridos por acidente no turno do DIA (`{xbar_dia_str}`) "
                        rf"é **estatisticamente {rel_amostra}** à do turno da NOITE (`{xbar_noite_str}`) ao nível de {conf_str} de confiança (|Z| = {z_stat_str} > {z_crit_str}).".replace(",", ".")
                    )
                else:
                    st.warning(
                        rf"**Conclusão da Inferência Amostral: Não se rejeita $H_0$.** "
                        rf"Com base na amostra ($n = {n_total:,}$ registros válidos), não há evidência estatística suficiente ao nível de {conf_str} "
                        rf"para afirmar que a média de feridos durante o dia (`{xbar_dia_str}`) difira da média no período da noite (`{xbar_noite_str}`) (|Z| = {z_stat_str} <= {z_crit_str}).".replace(",", ".")
                    )
            else:
                st.warning("A amostra atual não contém observações suficientes em ambos os turnos (Dia e Noite) para realizar o teste estatístico.")
        else:
            st.info("💡 *Nenhuma amostra foi gerada até o momento. Acesse a página '🧪 Amostra' e gere uma amostra aleatória para visualizar o teste inferencial.*")

        # ---------------------------------------------------------
        # 3. PARÂMETROS REAIS DA POPULAÇÃO (Ground Truth)
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🌐 2. Parâmetros Reais da População (Censo / Ground Truth)")
        st.caption(r"Como estes dados compreendem a totalidade dos registros da população selecionada, os valores abaixo representam as médias populacionais exatas ($\mu$), dispensando intervalos de confiança ou testes de hipótese probabilísticos.")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric(
                label="Média Real - Turno DIA (μ_dia)",
                value=f"{mu_dia_pop:.3f}".replace(".", ","),
                help=f"{df_pop_dia['feridos'].sum():,.0f} feridos em {N_pop_dia:,} acidentes (Desvio-Padrão: {sigma_dia_pop:.2f})".replace(",", ".")
            )
        with col_p2:
            st.metric(
                label="Média Real - Turno NOITE (μ_noite)",
                value=f"{mu_noite_pop:.3f}".replace(".", ","),
                help=f"{df_pop_noite['feridos'].sum():,.0f} feridos em {N_pop_noite:,} acidentes (Desvio-Padrão: {sigma_noite_pop:.2f})".replace(",", ".")
            )
        with col_p3:
            st.metric(
                label="Diferença Real Populacional (μ_dia - μ_noite)",
                value=f"{diff_real_mu:.3f}".replace(".", ",") + " feridos/acid.",
                help="Diferença determinística observada na população."
            )

        # ---------------------------------------------------------
        # 4. VALIDAÇÃO E INTERPRETAÇÃO DA INFERÊNCIA
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### ⚖️ 3. Avaliação e Validação da Inferência Amostral")

        if "df_amostrado" in st.session_state and n_dia >= 2 and n_noite >= 2:
            dia_coberto = ic_dia_inf <= mu_dia_pop <= ic_dia_sup
            noite_coberto = ic_noite_inf <= mu_noite_pop <= ic_noite_sup

            df_grafico_val = pd.DataFrame([
                {
                    "categoria": "Turno DIA",
                    "xbar": xbar_dia,
                    "ic_inf": ic_dia_inf,
                    "ic_sup": ic_dia_sup,
                    "mu_real": mu_dia_pop,
                    "rotulo_amostra": f"{xbar_dia:.3f} (IC {conf_str}: [{ic_dia_inf:.3f} - {ic_dia_sup:.3f}])".replace(".", ",")
                },
                {
                    "categoria": "Turno NOITE",
                    "xbar": xbar_noite,
                    "ic_inf": ic_noite_inf,
                    "ic_sup": ic_noite_sup,
                    "mu_real": mu_noite_pop,
                    "rotulo_amostra": f"{xbar_noite:.3f} (IC {conf_str}: [{ic_noite_inf:.3f} - {ic_noite_sup:.3f}])".replace(".", ",")
                }
            ])

            # Barra do IC da Amostra
            barras_amostra = (
                alt.Chart(df_grafico_val)
                .mark_rule(strokeWidth=4, color="#de2d26")
                .encode(
                    x=alt.X(
                        "ic_inf:Q",
                        title="Média de Feridos por Acidente",
                        scale=alt.Scale(
                            domain=[
                                max(0, min(df_grafico_val["ic_inf"].min(), df_grafico_val["mu_real"].min()) * 0.85),
                                max(df_grafico_val["ic_sup"].max(), df_grafico_val["mu_real"].max()) * 1.25
                            ]
                        )
                    ),
                    x2="ic_sup:Q",
                    y=alt.Y("categoria:N", title="Turno do Acidente", sort=["Turno DIA", "Turno NOITE"])
                )
            )

            # Ponto da estimativa pontual da amostra (\bar{x})
            pontos_amostra = (
                alt.Chart(df_grafico_val)
                .mark_point(size=130, filled=True, color="#de2d26")
                .encode(
                    x="xbar:Q",
                    y=alt.Y("categoria:N", sort=["Turno DIA", "Turno NOITE"]),
                    tooltip=[
                        alt.Tooltip("categoria:N", title="Turno"),
                        alt.Tooltip("xbar:Q", title="Média Amostral", format=".3f"),
                        alt.Tooltip("ic_inf:Q", title=f"IC Inf ({conf_str})", format=".3f"),
                        alt.Tooltip("ic_sup:Q", title=f"IC Sup ({conf_str})", format=".3f")
                    ]
                )
            )

            # Traço do Parâmetro Real da População (Ground Truth)
            pontos_real = (
                alt.Chart(df_grafico_val)
                .mark_tick(size=25, thickness=4, color="#1b7837")
                .encode(
                    x="mu_real:Q",
                    y=alt.Y("categoria:N", sort=["Turno DIA", "Turno NOITE"]),
                    tooltip=[
                        alt.Tooltip("categoria:N", title="Turno"),
                        alt.Tooltip("mu_real:Q", title="Média Real População (μ)", format=".3f")
                    ]
                )
            )

            # Rótulo em texto com a média amostral e o IC
            texto_amostra = (
                alt.Chart(df_grafico_val)
                .mark_text(align="left", baseline="middle", dx=12, color="#333333")
                .encode(
                    x="ic_sup:Q",
                    y=alt.Y("categoria:N", sort=["Turno DIA", "Turno NOITE"]),
                    text="rotulo_amostra:N"
                )
            )

            grafico_validacao = (
                (barras_amostra + pontos_amostra + pontos_real + texto_amostra)
                .properties(
                    height=200,
                    title=f"Validação: Estimativa Pontual e IC {conf_str} da Amostra (Vermelho) vs. Média Real Populacional (Traço Verde)"
                )
            )

            st.altair_chart(grafico_validacao, width="stretch")

            # Tabela de validação com cabeçalho dinâmico
            df_tabela_validacao = pd.DataFrame([
                {
                    "Turno": "DIA",
                    "Amostra (x̄)": f"{xbar_dia:.3f}".replace(".", ","),
                    f"IC {conf_str} da Amostra": f"[{ic_dia_inf:.3f}, {ic_dia_sup:.3f}]".replace(".", ","),
                    "Média Real População (μ)": f"{mu_dia_pop:.3f}".replace(".", ","),
                    "O Parâmetro Real está no IC?": "✅ Sim" if dia_coberto else "❌ Não"
                },
                {
                    "Turno": "NOITE",
                    "Amostra (x̄)": f"{xbar_noite:.3f}".replace(".", ","),
                    f"IC {conf_str} da Amostra": f"[{ic_noite_inf:.3f}, {ic_noite_sup:.3f}]".replace(".", ","),
                    "Média Real População (μ)": f"{mu_noite_pop:.3f}".replace(".", ","),
                    "O Parâmetro Real está no IC?": "✅ Sim" if noite_coberto else "❌ Não"
                }
            ])

            st.dataframe(df_tabela_validacao, hide_index=True, width="stretch")

            # Conclusão metodológica dinâmica
            N_pop_txt = f"{N_pop_total}".replace(",", ".")
            mu_dia_pop_txt = f"{mu_dia_pop:.3f}".replace(".", ",")
            mu_noite_pop_txt = f"{mu_noite_pop:.3f}".replace(".", ",")
            diff_real_txt = f"{abs(diff_real_mu):.3f}".replace(".", ",")
            sigma_dia_pop_txt = f"{sigma_dia_pop:.2f}".replace(".", ",")
            sigma_noite_pop_txt = f"{sigma_noite_pop:.2f}".replace(".", ",")
            rel_real = "superior" if mu_dia_pop > mu_noite_pop else "inferior"
            complemento_alfa = f"{100 - grau_conf_val}%"

            # Avalia se a decisão da amostra corresponde ao que de fato ocorre na população
            # Considera relevância prática para diferença de médias na população
            diferenca_pratica_pop = abs(diff_real_mu) >= 0.05

            if rejeita_h0 and diferenca_pratica_pop:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica:** A amostragem aleatória com nível de confiança de {conf_str} demonstrou alta eficácia, "
                    f"permitindo detectar a diferença significativa entre as médias de feridos por turno (|Z| = {z_stat_str} > {z_crit_str}) "
                    f"sem a necessidade de analisar a totalidade dos dados."
                )
            elif not rejeita_h0 and diferenca_pratica_pop:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica (Alerta de Poder Amostral):** A amostra não rejeitou a hipótese nula $H_0$ (|Z| = {z_stat_str} <= {z_crit_str}), "
                    f"enquanto na população completa há uma diferença substantiva de {diff_real_txt} feridos por acidente. "
                    f"Esse comportamento ilustra a ocorrência do **Erro Tipo II (falso negativo)** decorrente da variabilidade interna da contagem de feridos "
                    f"e de um tamanho amostral que limita o poder estatístico para detectar pequenas variações de médias."
                )
            else:
                conclusao_metodologica = (
                    f"**Conclusão Metodológica:** A inferência amostral indicou proximidade entre as médias de feridos nos dois turnos (|Z| = {z_stat_str} <= {z_crit_str}), "
                    f"mostrando plena consistência com os parâmetros reais observados na população de acidentes."
                )
                
            st.markdown(
                rf"""
                * **Verdade Factual da População:** Na população completa (**{N_pop_txt}** registros válidos), a média real de feridos por acidente durante o DIA é de **{mu_dia_pop_txt}** (desvio $\sigma$ = **{sigma_dia_pop_txt}**), enquanto durante a NOITE é de **{mu_noite_pop_txt}** (desvio $\sigma$ = **{sigma_noite_pop_txt}**). Portanto, na realidade do censo, a severidade média de feridos no dia é factual e determinística **{rel_real}** à da noite por **{diff_real_txt} feridos/acidente**.
                * **Eficácia da Cobertura:** {'Ambos os intervalos de confiança de ' + conf_str + ' calculados a partir da amostra capturaram as médias populacionais verdadeiras com sucesso.' if dia_coberto and noite_coberto else 'Houve desvio amostral em pelo menos um dos turnos em relação ao intervalo de ' + conf_str + ', o que é estatisticamente esperado em até ' + complemento_alfa + ' das amostras possíveis.'}
                * {conclusao_metodologica}
                """
            )
            
        else:
            N_pop_txt = f"{N_pop_total}".replace(",", ".")
            mu_dia_pop_txt = f"{mu_dia_pop:.3f}".replace(".", ",")
            mu_noite_pop_txt = f"{mu_noite_pop:.3f}".replace(".", ",")
            st.markdown(
                f"""
                * Na população completa (**{N_pop_txt}** registros), a média real de feridos por acidente durante o DIA é de **{mu_dia_pop_txt}**, enquanto à NOITE é de **{mu_noite_pop_txt}**.
                * Gere uma amostra na aba **'🧪 Amostra'** para confrontar as estimativas amostrais com os parâmetros reais da população.
                """
            )

