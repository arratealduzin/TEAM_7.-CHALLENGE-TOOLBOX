"""
Toolbox de Machine Learning para el análisis y selección de features.

Este módulo contiene funciones auxiliares para realizar análisis descriptivo,
tipificación de variables, selección de features numéricas y categóricas,
y visualización de relaciones para problemas de regresión en Machine Learning. 
Esto nos ayudará a preparar y entender mejor los datos antes de entrenar modelos 
predictivos durante el proceso de Machine Learning dentro del bootcamp, proyectos futuros y
más adelante en la carrera profesional.

Autor: Team T-07
Funciones:
    1. describe_df: Genera estadísticas descriptivas por columna
    2. tipifica_variables: Sugiere el tipo de cada variable
    3. get_features_num_regression: Selecciona features numéricas para regresión
    4. plot_features_num_regression: Visualiza relaciones numéricas
    5. get_features_cat_regression: Selecciona features categóricas para regresión
    6. plot_features_cat_regression: Visualiza relaciones categóricas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway
from typing import List, Optional


# ============================================================================
# FUNCIÓN 1: describe_df
# ============================================================================

def describe_df(df):
    """
    Genera un dataframe con estadísticas descriptivas de cada columna.
    
    Esta función analiza un dataframe y proporciona información sobre el tipo
    de dato, porcentaje de valores nulos, número de valores únicos y 
    cardinalidad porcentual de cada columna.
    
    Argumentos:
    -----------
    df (pd.DataFrame): 
        DataFrame a analizar.
    
    Devuelve:
    ---------
    pd.DataFrame: 
        DataFrame con las siguientes columnas:
        - data_type: Tipo de dato ('categorica' o 'numerica')
        - valores_nulos: Porcentaje de valores nulos (redondeado a 1 decimal)
        - valores_unicos: Número de valores únicos en la columna
        - cardinalidad_ptg: Porcentaje de cardinalidad (redondeado a 2 decimales)
    """
    
    # Diccionario para almacenar la información de cada columna
    data_dict = {}
    
    # Iterar sobre cada columna del dataframe
    for columna in df.columns:
        # Determinar si la columna es categórica o numérica
        # Si el tipo de dato es 'object', se considera categórica; en caso contrario, numérica
        data_type = "categorica" if df[columna].dtype == "object" else "numerica"
        
        # Calcular porcentaje de valores nulos redondeado a 1 decimal
        # Contar valores NaN, dividir por el total de filas y multiplicar por 100
        valores_nulos = round(df[columna].isna().sum() / len(df) * 100, 1)
        
        # Contar valores únicos (dropna=False para incluir NaN si es necesario)
        valores_unicos = df[columna].nunique()
        
        # Calcular cardinalidad en porcentaje redondeado a 2 decimales
        # Proporción de valores únicos respecto al total de filas
        cardinalidad_ptg = round(valores_unicos / len(df) * 100, 2)
        
        # Guardar información en el diccionario
        data_dict[columna] = {
            "data_type": data_type,
            "valores_nulos": valores_nulos,
            "valores_unicos": valores_unicos,
            "cardinalidad_ptg": cardinalidad_ptg
        }
    
    # Convertir diccionario a dataframe y transponer para tener columnas originales como índice
    return pd.DataFrame(data_dict).T


# ============================================================================
# FUNCIÓN 2: tipifica_variables
# ============================================================================

def tipifica_variables(df: pd.DataFrame, umbral_categoria: int, umbral_continua: float):
    """
    Sugiere el tipo de cada variable de un DataFrame en función de su cardinalidad.
    
    Esta función clasifica automáticamente las variables como Binaria, Categórica,
    Numérica Continua o Numérica Discreta basándose en su cardinalidad y en los
    umbrales proporcionados.
    
    Reglas de clasificación:
    - Cardinalidad = 2: "Binaria"
    - Cardinalidad < umbral_categoria: "Categórica"
    - Cardinalidad >= umbral_categoria:
        * Si % cardinalidad >= umbral_continua: "Numerica Continua"
        * En caso contrario: "Numerica Discreta"
    
    Argumentos:
    -----------
    df (pd.DataFrame): 
        DataFrame de entrada.
    umbral_categoria (int): 
        Umbral de cardinalidad para considerar una variable como "Categórica".
        Debe ser un entero positivo.
    umbral_continua (float): 
        Umbral (rango 0-1) del porcentaje de cardinalidad para considerar
        "Numérica Continua". Debe estar entre 0 y 1.
    
    Devuelve:
    ---------
    pd.DataFrame: 
        DataFrame con columnas ["nombre_variable", "tipo_sugerido"].
        Tendrá tantas filas como columnas tenga el dataframe de entrada.
    
    Raises:
    -------
    TypeError: 
        Si df no es un DataFrame de pandas.
    ValueError: 
        Si umbral_categoria no es un entero positivo o si umbral_continua
        no está en el rango [0, 1].
    """
    
    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un DataFrame de pandas")
    
    if not isinstance(umbral_categoria, int) or umbral_categoria <= 0:
        raise ValueError("umbral_categoria debe ser un entero positivo")
    
    if not isinstance(umbral_continua, (float, int)) or not (0 <= float(umbral_continua) <= 1):
        raise ValueError("umbral_continua debe ser un float entre 0 y 1")

    # Obtener el número de filas para calcular cardinalidad
    n_filas = len(df)
    
    # Lista para almacenar los resultados
    resultados = []

    # Iterar sobre cada columna del dataframe
    for col in df.columns:
        # Calcular cardinalidad (número de valores únicos, excluyendo NaN)
        card = df[col].nunique(dropna=True)

        # Calcular porcentaje de cardinalidad (proporción de valores únicos)
        pct_card = card / n_filas

        # Aplicar las reglas de clasificación
        if card == 2:
            # Si tiene exactamente 2 valores únicos, es binaria
            tipo = "Binaria"
        elif card < umbral_categoria:
            # Si cardinalidad es menor que el umbral, es categórica
            tipo = "Categórica"
        else:
            # Si cardinalidad es mayor o igual al umbral
            if pct_card >= float(umbral_continua):
                # Si el % de cardinalidad es alto, es numérica continua
                tipo = "Numerica Continua"
            else:
                # Si el % de cardinalidad es bajo, es numérica discreta
                tipo = "Numerica Discreta"

        # Añadir resultado a la lista
        resultados.append({"nombre_variable": col, "tipo_sugerido": tipo})

    # Convertir lista de resultados a dataframe
    return pd.DataFrame(resultados)


# ============================================================================
# FUNCIÓN 3: get_features_num_regression
# ============================================================================

def get_features_num_regression(df, target_col, umbral_corr, pvalue_entrada=None):
    """
    Obtiene variables numéricas que tienen una correlación significativa con 
    la variable target para un problema de regresión.
    
    Esta función identifica todas las columnas numéricas del dataframe cuya
    correlación de Pearson con la variable target supera un umbral especificado.
    Opcionalmente, puede filtrar aún más según la significancia estadística.
    
    Argumentos:
    -----------
    df (pd.DataFrame): 
        DataFrame que contiene las variables y el target.
    target_col (str): 
        Nombre de la columna target. Debe ser una variable numérica.
    umbral_corr (float): 
        Umbral de correlación absoluta (rango 0-1). Solo se incluyen variables
        cuya correlación absoluta con target_col sea superior a este valor.
    pvalue_entrada (float, opcional): 
        Umbral de p-valor para considerar la correlación como estadísticamente
        significativa. Si es None (por defecto), no se aplica filtro de significancia.
        Si se proporciona, solo se incluyen variables con p-valor < pvalue_entrada.
    
    Devuelve:
    ---------
    list: 
        Lista con los nombres de las columnas numéricas que cumplen los criterios
        de correlación y (si aplica) significancia estadística.
    None: 
        Si los argumentos de entrada no son válidos. Imprime un mensaje
        explicando el error.
    
    Validaciones:
    - umbral_corr debe estar entre 0 y 1
    - target_col debe existir en el dataframe
    - target_col debe ser de tipo numérico
    - pvalue_entrada debe ser None o un valor entre 0 y 1
    """
    
    # Validar que umbral_corr sea un número en el rango [0, 1]
    if not isinstance(umbral_corr, (int, float)) or not (0 < umbral_corr < 1):
        print("El umbral de correlación debe ser un número entre 0 y 1.")
        return None
    
    # Validar que target_col exista en el dataframe
    if target_col not in df.columns:
        print("La columna target no se encuentra en el DataFrame.")
        return None

    # Validar que target_col sea de tipo numérico
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("La columna target debe ser numérica (continua o discreta de alta cardinalidad) para regresión.")
        return None
    
    # Validar pvalue_entrada si no es None
    if pvalue_entrada is not None:
        if not isinstance(pvalue_entrada, (int, float)) or not (0 < pvalue_entrada <= 1):
            print("pvalue_entrada debe ser None o un número en (0, 1].")
            return None

    # Seleccionar todas las columnas numéricas del dataframe
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Eliminar la columna target de la lista de candidatas
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    # Lista para almacenar las columnas que cumplen los criterios
    lista_col_num = []
    
    # Iterar sobre cada columna numérica (excepto target_col)
    for col in num_cols:
        # Calcular correlación de Pearson y p-valor entre la columna y el target
        # pearsonr retorna (coeficiente_correlación, p_valor)
        corr, pvalue_calculado = pearsonr(df[col], df[target_col])
        
        # Verificar si la correlación absoluta supera el umbral
        if abs(corr) > umbral_corr:
            # Si pvalue_entrada es None, agregar directamente
            if pvalue_entrada is None:
                lista_col_num.append(col)
            # Si pvalue_entrada es especificado, verificar la significancia
            elif pvalue_calculado < pvalue_entrada:
                lista_col_num.append(col)
    
    # Retornar la lista de columnas que cumplen los criterios
    return lista_col_num


# ============================================================================
# FUNCIÓN 4: plot_features_num_regression
# ============================================================================

def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: Optional[List[str]] = None,
    umbral_corr: float = 0.0,
    pvalue: Optional[float] = None
) -> Optional[List[str]]:
    """
    Visualiza mediante pairplots las relaciones numéricas con el target en 
    problemas de regresión.
    
    Esta función genera pairplots para explorar la relación entre la variable
    target y otras variables numéricas seleccionadas. Si hay muchas variables,
    crea múltiples pairplots con un máximo de 5 columnas cada uno (incluida
    la variable target).
    
    Comportamiento:
    - Si 'columns' está vacía o es None: considera todas las variables numéricas
    - Si 'columns' no está vacía: filtra esa lista
    - En ambos casos: aplica filtros de correlación y p-valor si se especifican
    
    Argumentos:
    -----------
    df (pd.DataFrame): 
        DataFrame de entrada con los datos.
    target_col (str): 
        Nombre de la columna target. Debe ser una variable numérica.
        Por defecto es una cadena vacía.
    columns (list[str], opcional): 
        Lista de nombres de columnas candidatas a incluir en el análisis.
        Si es None o [], se consideran todas las variables numéricas.
        Por defecto es None.
    umbral_corr (float): 
        Umbral de correlación absoluta en el rango [0, 1].
        Solo se incluyen variables cuya |correlación| >= umbral_corr.
        Por defecto es 0.0.
    pvalue (float, opcional): 
        Umbral de significancia estadística en el rango (0, 1].
        Solo se incluyen variables con p-valor <= pvalue.
        Si es None, no se aplica filtro de significancia.
        Por defecto es None.
    
    Devuelve:
    ---------
    list[str]: 
        Lista de nombres de columnas que cumplen los criterios de correlación
        y significancia, y para las cuales se han generado pairplots.
    None: 
        Si los argumentos de entrada no son válidos. Imprime un mensaje
        explicando el error.
    []: 
        Lista vacía si no hay columnas que cumplan los criterios.
    
    Validaciones:
    - df debe ser un pandas DataFrame
    - target_col debe ser una cadena no vacía y existir en df
    - target_col debe ser numérica
    - umbral_corr debe estar en [0, 1]
    - pvalue debe ser None o estar en (0, 1]
    - columns debe ser una lista de strings o None
    
    Nota:
    -----
    EXTRA: Si hay muchas variables seleccionadas, genera varios pairplots
    con un máximo de 5 columnas por gráfico (target + 4 features).
    """

    # --------
    # Validaciones de entrada
    # --------
    
    # Validar que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un pandas DataFrame.")
        return None

    # Validar que target_col es una cadena no vacía
    if not isinstance(target_col, str) or target_col.strip() == "":
        print("Error: 'target_col' debe ser un string no vacío.")
        return None

    # Validar que target_col existe en el dataframe
    if target_col not in df.columns:
        print(f"Error: 'target_col' ({target_col}) no existe en el dataframe.")
        return None

    # Validar que target_col es numérico
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: 'target_col' debe ser una variable numérica.")
        return None

    # Validar que umbral_corr está en [0, 1]
    if not isinstance(umbral_corr, (int, float)) or not (0.0 <= float(umbral_corr) <= 1.0):
        print("Error: 'umbral_corr' debe estar en el rango [0, 1].")
        return None

    # Validar pvalue si no es None
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0.0 < float(pvalue) <= 1.0):
            print("Error: 'pvalue' debe ser None o un float en (0, 1].")
            return None

    # Validar columns y convertir a lista si es None
    if columns is None:
        columns = []
    
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        print("Error: 'columns' debe ser una lista de strings o None.")
        return None

    # --------
    # Determinar columnas candidatas
    # --------
    
    # Obtener todas las columnas numéricas del dataframe
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Si columns está vacía, usar todas las numéricas (excepto target)
    if len(columns) == 0:
        candidate_cols = [c for c in numeric_cols if c != target_col]
    else:
        # Si columns no está vacía, filtrar y validar
        # Advertir sobre columnas inexistentes
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"Aviso: se ignoran columnas que no existen: {missing}")

        # Mantener solo columnas numéricas existentes y distintas de target
        candidate_cols = [c for c in columns if c in df.columns and c != target_col]
        candidate_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

    # Retornar lista vacía si no hay columnas candidatas
    if len(candidate_cols) == 0:
        print("No hay columnas numéricas candidatas para evaluar.")
        return []

    # --------
    # Filtrado por correlación y p-valor
    # --------
    
    selected = []
    
    # Iterar sobre cada columna candidata
    for c in candidate_cols:
        # Eliminar filas con valores NaN en target o en la columna
        valid = df[[target_col, c]].dropna()

        # Validar que hay suficientes datos (mínimo 3 observaciones)
        if valid.shape[0] < 3:
            continue
        
        # Validar que ambas variables tienen variación (al menos 2 valores únicos)
        if valid[target_col].nunique() < 2 or valid[c].nunique() < 2:
            continue

        # Calcular correlación de Pearson
        r, p = pearsonr(valid[target_col], valid[c])

        # Aplicar criterios: correlación absoluta >= umbral_corr
        if abs(r) >= float(umbral_corr):
            # Si pvalue es None, incluir directamente
            # Si pvalue es especificado, verificar significancia
            if pvalue is None or p <= float(pvalue):
                selected.append(c)

    # Retornar lista vacía si ninguna variable cumple los criterios
    if len(selected) == 0:
        print("Ninguna columna cumple los criterios de correlación / significación.")
        return []

    # --------
    # Generar pairplots (máximo 5 columnas por plot)
    # --------
    
    # Máximo de columnas por pairplot (incluyendo target)
    max_cols_per_plot = 5
    
    # Máximo de features adicionales por plot (5 - 1 para el target)
    chunk_size = max_cols_per_plot - 1

    # Iterar sobre chunks de features para crear múltiples pairplots si es necesario
    for i in range(0, len(selected), chunk_size):
        # Obtener un chunk de features
        chunk = selected[i:i + chunk_size]
        
        # Crear lista de columnas a plotear (target + features del chunk)
        cols_to_plot = [target_col] + chunk

        # Eliminar filas con NaN en cualquiera de las columnas a plotear
        plot_df = df[cols_to_plot].dropna()
        
        # Validar que hay suficientes datos para el pairplot
        if plot_df.shape[0] < 3:
            continue

        # Crear y mostrar el pairplot
        sns.pairplot(plot_df, corner=True)
        
        # Añadir título al gráfico
        plt.suptitle(f"Pairplot: {target_col} vs {chunk}", y=1.02)
        plt.show()

    # Retornar la lista de columnas seleccionadas
    return selected


# ============================================================================
# FUNCIÓN 5: get_features_cat_regression
# ============================================================================

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Devuelve las variables categóricas que tienen relación estadística 
    significativa con una variable target numérica para un problema de regresión.
    
    Esta función realiza un análisis de varianza (ANOVA) unidireccional para
    cada variable categórica, evaluando si existen diferencias significativas
    en la distribución de la variable target entre los grupos definidos por
    cada categoría.
    
    Argumentos:
    -----------
    df (pd.DataFrame): 
        DataFrame con los datos.
    target_col (str): 
        Nombre de la columna target. Debe ser una variable numérica continua
        o discreta de alta cardinalidad.
    pvalue (float): 
        Umbral de significación para el test ANOVA. Por defecto es 0.05.
        Solo se incluyen variables categóricas con p-valor < pvalue.
    
    Devuelve:
    ---------
    list: 
        Lista de nombres de columnas categóricas relacionadas estadísticamente
        con la variable target.
    None: 
        Si los argumentos de entrada no son válidos. Imprime un mensaje
        explicando el error.
    
    Validaciones:
    - df debe ser un pandas DataFrame
    - target_col debe existir en el dataframe y ser numérico
    - pvalue debe ser un float en (0, 1)
    - Debe haber al menos variables categóricas en el dataframe
    
    Nota:
    -----
    Se utiliza el test de ANOVA (f_oneway) que contrasta si las medias del
    target son iguales en todos los grupos definidos por la variable categórica.
    """

    # Validar que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("df no es un DataFrame")
        return None

    # Validar que target_col existe en el DataFrame
    if target_col not in df.columns:
        print("target_col no existe en el DataFrame")
        return None

    # Validar que target_col es numérico
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("target_col no es numérica")
        return None

    # Validar que pvalue es un float válido
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
        print("pvalue no es válido")
        return None

    # Obtener todas las columnas categóricas del dataframe
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Retornar None si no hay variables categóricas
    if len(cat_cols) == 0:
        print("No hay variables categóricas")
        return None

    # Lista para almacenar las columnas categóricas seleccionadas
    selected_cols = []

    # Iterar sobre cada variable categórica
    for col in cat_cols:
        # Crear una lista de grupos (arrays) para cada valor único de la categoría
        groups = []
        
        # Iterar sobre cada valor único en la columna categórica
        for value in df[col].dropna().unique():
            # Extraer los valores del target para este grupo y eliminar NaN
            group = df[df[col] == value][target_col].dropna()
            
            # Solo incluir el grupo si tiene al menos 1 observación
            if len(group) > 1:
                groups.append(group)

        # Saltar esta variable si tiene menos de 2 grupos
        if len(groups) < 2:
            continue

        # Realizar el test ANOVA
        try:
            # f_oneway realiza ANOVA y retorna (estadístico_F, p_valor)
            _, p = f_oneway(*groups)
            
            # Si el p-valor es menor que el umbral, incluir la variable
            if p < pvalue:
                selected_cols.append(col)
        except:
            # Si hay algún error en el cálculo, continuar con la siguiente variable
            continue

    # Retornar la lista de columnas categóricas seleccionadas
    return selected_cols


# ============================================================================
# FUNCIÓN 6: plot_features_cat_regression
# ============================================================================

def plot_features_cat_regression(df, target_col="", columns=None, pvalue=0.05, with_individual_plot=False):
    """
    Visualiza mediante histogramas agrupados las relaciones entre variables
    categóricas y la variable target numérica en problemas de regresión.
    
    Esta función muestra histogramas del target numérico, separados por los
    valores de cada variable categórica que tiene relación estadísticamente
    significativa con el target.
    
    Comportamiento:
    - Si 'columns' está vacía o es None: considera todas las variables categóricas
    - Si 'columns' no está vacía: filtra esa lista
    - En ambos casos: aplica filtro de significancia basado en ANOVA
    
    Argumentos:
    -----------
    df (pd.DataFrame): 
        DataFrame con los datos.
    target_col (str): 
        Nombre de la columna target. Debe ser una variable numérica.
        Por defecto es una cadena vacía.
    columns (list, opcional): 
        Lista de nombres de columnas categóricas a analizar.
        Si es None o [], se consideran todas las variables categóricas.
        Por defecto es None.
    pvalue (float): 
        Umbral de significación para el test ANOVA.
        Solo se incluyen variables con p-valor < pvalue.
        Por defecto es 0.05.
    with_individual_plot (bool): 
        Si es True, genera un histograma individual para cada variable categórica.
        Si es False, solo retorna la lista sin generar gráficos.
        Por defecto es False.
    
    Devuelve:
    ---------
    list: 
        Lista de nombres de columnas categóricas relacionadas estadísticamente
        con la variable target.
    None: 
        Si los argumentos de entrada no son válidos. Imprime un mensaje
        explicando el error.
    
    Validaciones:
    - df debe ser un pandas DataFrame
    - target_col debe ser una cadena no vacía y existir en df
    - target_col debe ser numérico
    - columns debe ser una lista de strings o None
    
    Nota:
    -----
    Si with_individual_plot es True, genera un histograma por cada variable
    categórica válida, mostrando la distribución del target para cada categoría
    con colores y transparencia diferentes para distinguir los grupos.
    """

    # Validar que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("df no es un DataFrame")
        return None

    # Validar que target_col existe en el DataFrame
    if target_col not in df.columns:
        print("target_col no es válido")
        return None

    # Validar que target_col es numérico
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("target_col no es numérica")
        return None

    # Si columns es None, consideramos todas las variables categóricas
    if columns is None or len(columns) == 0:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Lista para almacenar las columnas categóricas seleccionadas
    selected_cols = []

    # Iterar sobre cada variable categórica candidata
    for col in columns:
        # Saltar si la columna no existe en el dataframe
        if col not in df.columns:
            continue

        # Crear lista de grupos para cada valor único de la categoría
        groups = [
            df[df[col] == value][target_col].dropna()
            for value in df[col].dropna().unique()
        ]

        # Saltar si hay menos de 2 grupos
        if len(groups) < 2:
            continue

        # Realizar test ANOVA
        _, p = f_oneway(*groups)

        # Si p-valor es significativo, incluir en la lista
        if p < pvalue:
            selected_cols.append(col)

            # Si se solicita, crear gráficos individuales
            if with_individual_plot:
                # Crear una nueva figura
                plt.figure(figsize=(10, 6))
                
                # Iterar sobre cada valor único de la categoría
                for value in df[col].dropna().unique():
                    # Plotear histograma para cada grupo (con transparencia)
                    df[df[col] == value][target_col].hist(alpha=0.5, label=str(value), bins=20)
                
                # Añadir etiquetas y leyenda
                plt.title(f"{target_col} vs {col}", fontsize=14, fontweight='bold')
                plt.xlabel(target_col, fontsize=12)
                plt.ylabel("Frecuencia", fontsize=12)
                plt.legend(title=col)
                plt.grid(alpha=0.3)
                
                # Mostrar el gráfico
                plt.show()

    # Retornar la lista de columnas categóricas seleccionadas
    return selected_cols
