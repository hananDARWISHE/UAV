from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR # استدعاء SVR
from sklearn.multioutput import MultiOutputRegressor # لأن SVR لا يدعم 3 مخرجات مباشرة
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# ----------------- تجهيز البيانات وتدريب 3 نماذج -----------------
try:
    df = pd.read_csv('data.csv', encoding='utf-8-sig').dropna()
    df.columns = df.columns.str.strip()

    le_terrain = LabelEncoder()
    le_land = LabelEncoder()
    df['Terrain_Type_Num'] = le_terrain.fit_transform(df['Terrain_Type'].astype(str))
    df['Land_Cover_Num'] = le_land.fit_transform(df['Land_Cover'].astype(str))

    features = ['Point_Elevation (m)', 'Flight_Altitude (m)', 'GSD (cm)', 'Flight_Speed (m/s)', 
                'Nadir_Angle (deg)', 'Overlap_Front (%)', 'Overlap_Side (%)', 'Terrain_Type_Num', 'Land_Cover_Num']
    X = df[features]
    y = df[['Error_X (cm)', 'Error_Y (cm)', 'Error_Z (cm)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

    # 2. ANN
    ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42).fit(X_scaled, y)

    # 3. SVR (Support Vector Regression)
    # ملاحظة: SVR يحتاج لمغلف MultiOutput للتعامل مع X, Y, Z معاً
    svr_model = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1)).fit(X_scaled, y)

    # حساب الدقة لكل نموذج
    rf_score = round(rf_model.score(X, y) * 100, 2)
    ann_score = round(ann_model.score(X_scaled, y) * 100, 2)
    svr_score = round(svr_model.score(X_scaled, y) * 100, 2)

    print("✅ تم تدريب 3 نماذج بنجاح (RF, ANN, SVR)")

except Exception as e:
    print(f"❌ فشل التدريب: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            # استقبال البيانات
            inputs_raw = [
                float(request.form['ele']), float(request.form['alt']),
                (float(request.form['alt']) * float(request.form['sensor']) * 100) / (float(request.form['focal']) * float(request.form['img_width'])),
                4.0, 90.0, 80.0, 70.0, int(request.form['terrain']), int(request.form['land'])
            ]
            
            # التنبؤات
            rf_p = rf_model.predict([inputs_raw])[0]
            
            scaled_in = scaler.transform([inputs_raw])
            ann_p = ann_model.predict(scaled_in)[0]
            svr_p = svr_model.predict(scaled_in)[0]
            
            results = {
                'gsd': round(inputs_raw[2], 2),
                'rf': [round(x, 3) for x in rf_p],
                'ann': [round(x, 3) for x in ann_p],
                'svr': [round(x, 3) for x in svr_p],
                'rf_acc': rf_score, 'ann_acc': ann_score, 'svr_acc': svr_score
            }
        except Exception as e:
            results = f"خطأ: {e}"
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)