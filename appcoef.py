import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Titre de l'application
st.title('Analyse de Régression Linéaire Multiple')

# Téléchargement du fichier CSV ou Excel
uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=['csv', 'xls', 'xlsx'])

if uploaded_file is not None:
    # Lire le fichier en fonction de l'extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Affichage d'un aperçu des données
    st.write("Aperçu des données:", df.head())

    # Sélection de la colonne cible (variable dépendante)
    target_column = st.selectbox('Sélectionnez la colonne de la target (variable dépendante)', df.columns)

    # Sélection des colonnes des variables explicatives (indépendantes)
    feature_columns = st.multiselect('Sélectionnez les colonnes des variables explicatives (indépendantes)', df.columns)

    # Sélection de la taille de la fenêtre d'entraînement
    window_size = st.slider('Sélectionnez la taille de la fenêtre d\'entraînement (en nombre de lignes)', 2, len(df), 8)

    if target_column and feature_columns:
        # Fonction pour ajuster le modèle de régression linéaire sans intercept
        def ajuster_modele(X, y):
            model = sm.OLS(y, X).fit()
            return model.params, model.rsquared, model.predict(X)

        # Initialiser des listes pour stocker les coefficients, R-squared, dates, prédictions et valeurs réelles
        dates = []
        coeffs = {col: [] for col in feature_columns}
        r_squared = []
        predictions = []
        actual_values = []

        # Effectuer la régression avec une fenêtre mobile
        for start in range(len(df) - window_size):
            end = start + window_size
            X = df[feature_columns].iloc[start:end]
            y = df[target_column].iloc[start:end]
            params, rsquared, y_pred = ajuster_modele(X, y)

            dates.append(df.index[end])
            for col in feature_columns:
                coeffs[col].append(params[col])
            r_squared.append(rsquared)
            predictions.append(y_pred.iloc[-1])  # Stocker la dernière prédiction pour comparaison
            actual_values.append(y.iloc[-1])  # Stocker la dernière valeur réelle

        # Convertir les listes en DataFrame
        coef_df = pd.DataFrame(coeffs)
        coef_df['Date'] = dates
        coef_df['R-squared'] = r_squared

        pred_df = pd.DataFrame({
            'Date': dates,
            'Valeur Réelle': actual_values,
            'Valeur Prédite': predictions
        })

        # Affichage des résultats sous forme de tableau
        st.write("Évolution des coefficients et du R-squared:", coef_df)

        # Visualisation de la comparaison entre valeurs réelles et prédites
        st.write("Comparaison entre les valeurs réelles et les valeurs prédites")
        plt.figure(figsize=(10, 5))
        plt.plot(pred_df['Date'], pred_df['Valeur Réelle'], label='Valeur Réelle', color='blue')
        plt.plot(pred_df['Date'], pred_df['Valeur Prédite'], label='Valeur Prédite', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel(f'Valeur de {target_column}')
        plt.title(f'Comparaison entre les valeurs réelles et prédites pour {target_column}')
        plt.legend()
        st.pyplot(plt)

        # Visualisation de l'évolution des coefficients, un graphique par coefficient
        st.write("Graphique de l'évolution des coefficients")
        colors = plt.cm.get_cmap('tab10', len(feature_columns))  # Générer des couleurs

        for idx, col in enumerate(feature_columns):
            plt.figure(figsize=(10, 5))
            plt.plot(coef_df['Date'], coef_df[col], label=f'Coefficient de {col}', color=colors(idx))
            plt.xlabel('Date')
            plt.ylabel(f'Coefficient de {col}')
            plt.title(f'Évolution du coefficient de {col}')
            plt.legend()
            st.pyplot(plt)

        # Visualisation de l'évolution du R-squared
        st.write("Graphique de l'évolution du R-squared")
        plt.figure(figsize=(10, 5))
        plt.plot(coef_df['Date'], coef_df['R-squared'], label='R-squared', color='green')
        plt.xlabel('Date')
        plt.ylabel('R-squared')
        plt.title('Évolution du R-squared')
        plt.legend()
        st.pyplot(plt)
