import folium
import openrouteservice as ors
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# بيانات افتراضية لتدريب النموذج (المسافة، الوقت السابق للوصول، حالة المرور)
data = np.array([
   [5.0, 10.0, 1, 30.0],
   [3.0, 8.0, 0, 20.0],
   [6.0, 12.0, 1, 40.0],
   [2.0, 6.0, 0, 15.0],
   [4.0, 9.0, 1, 25.0],
   [7.0, 14.0, 0, 45.0],
   [3.5, 7.5, 1, 22.0],
   [5.5, 11.0, 0, 35.0],
])

#تقسيم البيانات إلى ميزات ومستهدف
X = data[:, :-1]
y = data[:, -1]
# تقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب نموذج Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# حساب الدقة باستخدام مجموعة الاختبار
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"Model accuracy (R^2 score): {accuracy:.2f}")

# إعداد إحداثيات المواقع (المواقع الافتراضية)
coords = [
    [-87.7898356, 41.8879452],
    [-87.7808524, 41.8906422],
    [-87.7895149, 41.8933762],
    [-87.7552925, 41.8809087],
    [-87.7728134, 41.8804058],
    [-87.7702890, 41.8802231],
    [-87.7787924, 41.8944518],
    [-87.7732345, 41.8770663],
]

# إعداد نقطة البداية
vehicle_start = [-87.800701, 41.876214]

# إنشاء الخريطة
m = folium.Map(location=list(reversed([-87.787984, 41.8871616])), tiles="cartodbpositron", zoom_start=14)

# إضافة النقاط على الخريطة
for coord in coords:
    folium.Marker(location=list(reversed(coord))).add_to(m)
    
folium.Marker(location=list(reversed(vehicle_start)), icon=folium.Icon(color="red")).add_to(m)

# إعداد عميل OpenRouteService بمفتاح الـ API الخاص بك
client = ors.Client(key='YOUR_API_KEY_HERE')  # استبدل 'YOUR_API_KEY_HERE' بمفتاح الـ API الخاص بك

# إعداد المركبات
vehicles = [
    ors.optimization.Vehicle(id=0, profile='driving-car', start=vehicle_start, end=vehicle_start, capacity=[5]),
    ors.optimization.Vehicle(id=1, profile='driving-car', start=vehicle_start, end=vehicle_start, capacity=[5])
]

# إعداد المهام (Jobs)
jobs = [ors.optimization.Job(id=index, location=coords, amount=[1]) for index, coords in enumerate(coords)]

# تحسين المسار باستخدام ORS
optimized = client.optimization(jobs=jobs, vehicles=vehicles, geometry=True)

# حساب المسافة الفعلية بين نقطة البداية وكل نقطة
def calculate_distance(coord1, coord2):
    matrix = client.distance_matrix(
        locations=[coord1, coord2],
        profile='driving-car',
        metrics=['distance']
    )
    return matrix['distances'][0][1] / 1000.0  # المسافة بالكيلومترات

# عرض المسارات المحسنة على الخريطة
line_colors = ['blue', 'orange', ' green', 'yellow']
for route in optimized['routes']:
    folium.PolyLine(locations=[list(reversed(coords)) for coords in ors.convert.decode_polyline(route['geometry'])['coordinates']], color=line_colors[route['vehicle']]).add_to(m)
    for step in route['steps']:
        if step['type'] == 'job':
            # حساب المسافة الفعلية
            distance = calculate_distance(vehicle_start, step['location'])
            # استخدام قيم عشوائية للوقت السابق وحالة المرور للتبسيط
            prev_time = np.random.uniform(5.0, 15.0)  # قيمة افتراضية للوقت السابق للوصول بالدقائق
            traffic_status = np.random.choice([0, 1])  # حالة المرور (0: منخفض، 1: عالي)
            
            # تنبؤ بوقت الوصول باستخدام النموذج المدرب
            input_data = np.array([distance, prev_time, traffic_status]).reshape(1, -1)
            predicted_time = model.predict(input_data)[0]
            folium.Marker(
                location=list(reversed(step['location'])), 
                popup=f"Distance: {distance:.2f} km<br>Predicted arrival time: {predicted_time:.2f} minutes", 
                icon=folium.Icon(color=line_colors[route['vehicle']])
            ).add_to(m)

# عرض الخريطة
m
