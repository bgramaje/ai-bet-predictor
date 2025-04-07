# %% 1. Librerías requeridas
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import poisson
import os
import warnings
warnings.filterwarnings('ignore')

# %% 2. Clase principal del predictor
class CornerPredictor:
    def __init__(self):
        self.model = None
        self.features = [
            'h_presion_ataque', 'AC_avg', 'h_corner_eff', 'h_intensidad',
            'a_presion_ataque', 'HC_avg', 'a_corner_eff', 'a_intensidad',
            'h_presion_x_a_vuln', 'a_presion_x_h_vuln'
        ]
        self.team_stats = None

    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("El archivo CSV está vacío")
                
            required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'HC', 'AC', 'HST', 'AST', 
                              'HS', 'AS', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise KeyError(f"Columnas faltantes: {missing}")
                
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date'])
            return df.sort_values('Date').reset_index(drop=True)
            
        except Exception as e:
            print(f"Error cargando datos: {str(e)}")
            return None
    
    def preprocess(self, df):
        # Paso 1: Manejar valores faltantes
        numeric_cols = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
        df[numeric_cols] = df[numeric_cols].fillna(0).astype(float)
        
        # Paso 2: Calcular estadísticas de equipos
        self.team_stats = df.groupby('HomeTeam').agg({
            'HC': 'mean',
            'AC': 'mean',
            'HST': 'mean',
            'HS': 'mean',
            'HF': 'mean',
            'HY': 'mean',
            'HR': 'mean'
        }).reset_index()
        
        # Paso 3: Medias móviles adaptativas
        for equipo in df['HomeTeam'].unique():
            home_mask = df['HomeTeam'] == equipo
            away_mask = df['AwayTeam'] == equipo
            
            df.loc[home_mask, 'HC_avg'] = df.loc[home_mask, 'HC'].expanding(min_periods=1).mean()
            df.loc[away_mask, 'AC_avg'] = df.loc[away_mask, 'AC'].expanding(min_periods=1).mean()
        
        # Paso 4: Construir características
        df['h_presion_ataque'] = df['HST']
        df['a_presion_ataque'] = df['AST']
        df['h_corner_eff'] = np.where(
            df['HS'] > 0, 
            df['HC'] / df['HS'], 
            df['HC'] / 1  # Evitar división por cero
        )
        df['a_corner_eff'] = np.where(
            df['AS'] > 0,
            df['AC'] / df['AS'],
            df['AC'] / 1
        )
        df['h_intensidad'] = df['HF'] + df['HY']*0.3 + df['HR']*0.7
        df['a_intensidad'] = df['AF'] + df['AY']*0.3 + df['AR']*0.7
        df['h_presion_x_a_vuln'] = df['h_presion_ataque'] * df['AC_avg']
        df['a_presion_x_h_vuln'] = df['a_presion_ataque'] * df['HC_avg']
        
        # Paso 5: Eliminar filas inválidas
        return df.dropna(subset=self.features + ['HC', 'AC'])
    
    def train_model(self, df):
        if len(df) < 10:
            raise ValueError("Insuficientes datos para entrenar (mínimo 10 partidos)")
            
        preprocessor = ColumnTransformer(
            transformers=[('num', 'passthrough', self.features)]
        )
        
        self.model = Pipeline([
            ('prep', preprocessor),
            ('model', MultiOutputRegressor(
                XGBRegressor(
                    objective='count:poisson',
                    n_estimators=500,
                    learning_rate=0.07,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.7
                )
            ))
        ])
        
        self.model.fit(df[self.features], df[['HC', 'AC']])
    
    def predict_match(self, home_team, away_team):
        try:
            # Obtener estadísticas de equipos
            home_stats = self.team_stats[self.team_stats['HomeTeam'] == home_team].iloc[0]
            away_stats = self.team_stats[self.team_stats['HomeTeam'] == away_team].iloc[0]
            
            # Construir datos de predicción
            match_data = {
                'h_presion_ataque': home_stats['HST'],
                'AC_avg': away_stats['AC'],
                'h_corner_eff': home_stats['HC'] / (home_stats['HS'] + 1e-5),
                'h_intensidad': home_stats['HF'] + home_stats['HY']*0.3 + home_stats['HR']*0.7,
                'a_presion_ataque': away_stats['HST'],
                'HC_avg': home_stats['HC'],
                'a_corner_eff': away_stats['AC'] / (away_stats['HS'] + 1e-5),
                'a_intensidad': away_stats['HF'] + away_stats['HY']*0.3 + away_stats['HR']*0.7,
                'h_presion_x_a_vuln': home_stats['HST'] * away_stats['AC'],
                'a_presion_x_h_vuln': away_stats['HST'] * home_stats['HC']
            }
            
            # Predicción con el modelo entrenado
            prediction_df = pd.DataFrame([match_data])
            prediction = self.model.predict(prediction_df[self.features])
            
            # Calcular distribuciones Poisson
            hc_pred = max(0, prediction[0][0])
            ac_pred = max(0, prediction[0][1])
            
            return {
                home_team: {
                    "Predicción": round(hc_pred, 1),
                    "Máximo probable": int(np.argmax(poisson.pmf(np.arange(15), hc_pred))),
                    "Intervalo 70%": f"{max(0, int(hc_pred-1))}-{int(hc_pred+1)}"
                },
                away_team: {
                    "Predicción": round(ac_pred, 1),
                    "Máximo probable": int(np.argmax(poisson.pmf(np.arange(15), ac_pred))),
                    "Intervalo 70%": f"{max(0, int(ac_pred-1))}-{int(ac_pred+1)}"
                }
            }
            
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
    
# %% 3. Ejecución principal del script
if __name__ == "__main__":
    predictor = CornerPredictor()
    
    # Cargar datos desde el archivo CSV ubicado en la carpeta "data"
    data_path = "data/season-2425.csv"
    data_frame = predictor.load_data(data_path)
    
    if data_frame is None or data_frame.empty:
        print("No se pudieron cargar los datos. Verifica:")
        print(f"1. Que el archivo {data_path} existe")
        print("2. Que las columnas requeridas están presentes")
        print("3. Que hay al menos 10 partidos válidos")
        exit()
    
    # Preprocesar los datos cargados y entrenar el modelo
    try:
        processed_data_frame = predictor.preprocess(data_frame)
        
        if processed_data_frame.empty:
            raise ValueError("Datos preprocesados están vacíos")
        
        predictor.train_model(processed_data_frame)
        
    except Exception as e:
        print(f"Error durante el preprocesamiento o entrenamiento: {str(e)}")
    
    print("\n🔥 Predictor de Córners - La Liga")
    print("Instrucciones:")
    print("- Ingresa partidos en formato Local - Visitante (ejemplo: Barcelona - Real Madrid)")
    print("- Escribe salir para terminar\n")
    
    while True:
        user_input = input("👉 Partido a predecir: ").strip()
        
        if user_input.lower() in ["salir", "exit"]:
            break
        
        if "-" not in user_input:
            print("❌ Formato incorrecto. Usa Local - Visitante")
        
        else:
            try:
                local_team, visiting_team = [team.strip() for team in user_input.split("-")]
                
                prediction_result = predictor.predict_match(local_team, visiting_team)
                
                if prediction_result:
                    print(f"\n⚽ {local_team} vs {visiting_team}")
                    
                    local_prediction = prediction_result[local_team]
                    visiting_prediction = prediction_result[visiting_team]
                    
                    print(f"📍 {local_team}: {local_prediction}")
                    print(f"📍 {visiting_team}: {visiting_prediction}")
                
                else:
                    print("❌ No se pudo realizar la predicción.")
            
            except Exception as e:
                print(f"❌ Error: {str(e)}\n")
