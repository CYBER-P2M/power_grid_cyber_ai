# power_grid_cyber_ai
Windows


# أوامر تشغيل مشروع PowerGrid Cyber AI

## 1. فتح مجلد المشروع

افتح Command Prompt أو PowerShell ونفّذ:

```cmd
cd power_grid_cyber_ai
```

تحقق من أنك داخل جذر المشروع:

```cmd
dir
```

يجب أن تظهر مجلدات مثل:

```text
data
models
results
src
app
dashboard
tools
requirements.txt
```

## 2. إنشاء البيئة الافتراضية

نفّذ هذه الخطوة مرة واحدة فقط إذا لم تكن البيئة موجودة:

```cmd
python -m venv .venv
```

## 3. تفعيل البيئة

في Command Prompt:

```cmd
.venv\Scripts\activate
```

في PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

يجب أن يظهر اسم البيئة في بداية السطر:

```text
(.venv)
```

إذا منع PowerShell تشغيل السكربتات، افتح PowerShell كمسؤول ونفّذ مرة واحدة:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

ثم فعّل البيئة:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4. تثبيت المتطلبات

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

لإنشاء نسخة EXE لاحقاً:

```cmd
pip install pyinstaller
```

## 5. التحقق من ملفات البيانات

إذا كانت البيانات داخل `data/raw`:

```cmd
dir data\raw
```

إذا كانت البيانات النهائية داخل `data/selected`:

```cmd
dir data\selected
```

استخدم نسخة واحدة متجانسة من Dataset، ولا تخلط ملفات `data1.csv` إلى `data15.csv` مع نسخ Kaggle أو ملفات Figshare التدريبية والاختبارية إلا بعد توثيق ذلك.

## 6. التحقق من عمود التسمية

لنسخة `data/raw`:

```cmd
python -c "import pandas as pd,glob; d=pd.concat([pd.read_csv(f,usecols=['marker']) for f in glob.glob('data/raw/*.csv')],ignore_index=True); print(d['marker'].value_counts(dropna=False).to_string())"
```

لنسخة `data/selected`:

```cmd
python -c "import pandas as pd,glob; d=pd.concat([pd.read_csv(f,usecols=['marker']) for f in glob.glob('data/selected/*.csv')],ignore_index=True); print(d['marker'].value_counts(dropna=False).to_string())"
```

## 7. تحديد مسار البيانات

افتح:

```text
src/config.py
```

إذا كانت البيانات داخل `data/raw` استخدم:

```python
RAW_DIR = ROOT / "data" / "raw"
```

إذا كانت البيانات النهائية داخل `data/selected` استخدم:

```python
RAW_DIR = ROOT / "data" / "selected"
```

لا تستخدم المسارين في الوقت نفسه.

## 8. تدريب النماذج

عمود التسمية في Dataset الحالية هو `marker`:

```cmd
python src\train_models.py --target marker
```

انتظر حتى تظهر:

```text
Training completed successfully.
```

ينشئ التدريب الملفات التالية:

```text
models\random_forest.joblib
models\xgboost_model.joblib
models\imputer.joblib
models\label_encoder.joblib
models\feature_order.json
models\model_metadata.json
results\predictions_test.csv
```

## 9. تقييم النماذج

```cmd
python src\evaluate_models.py
```

ينشئ:

```text
results\confusion_matrices\
results\classification_reports\
results\fault_attack_analysis.csv
```

تحقق من ملف أخطاء Fault وAttack:

```cmd
type results\fault_attack_analysis.csv
```

## 10. استخراج أهم الخصائص

```cmd
python src\explainability.py
```

ينشئ:

```text
results\top_20_features.csv
```

لعرضه:

```cmd
type results\top_20_features.csv
```

## 11. اختبار Risk Score من الطرفية

اختبار مباشر:

```cmd
python app\risk_score.py
```

إذا كان لديك ملف JSON لصف واحد:

```cmd
python app\risk_score.py --input app\sample_input.json
```

يجب أن يعرض:

```text
predicted_class
risk_score
probabilities
```

## 12. تشغيل Dashboard

شغّلها من جذر المشروع:

```cmd
python -m streamlit run dashboard\app.py
```

أو:

```cmd
streamlit run dashboard\app.py
```

افتح المتصفح على:

```text
http://localhost:8501
```

تحتوي Dashboard على:

```text
Overview
Live Risk
Evaluation
Feature Importance
Real Prediction
About
```

## 13. تشغيل صفحة Real Prediction

تأكد من وجود الصفحة:

```cmd
dir dashboard\pages
```

يجب أن يظهر:

```text
2_Real_Prediction.py
```

داخل الصفحة يجب أن يكون مسار الجذر:

```python
ROOT = Path(__file__).resolve().parents[2]
```

لا تشغّل الصفحة منفردة. شغّل دائماً التطبيق الرئيسي:

```cmd
python -m streamlit run dashboard\app.py
```

ثم افتح `2 Real Prediction` وارفع ملف CSV يحتوي على خصائص التدريب الـ128.

## 14. إيقاف Dashboard

داخل نافذة التشغيل اضغط:

```text
Ctrl + C
```
## 18. تسلسل التشغيل الكامل من الصفر

```cmd
cd power_grid_cyber_ai_full
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\train_models.py --target marker
python src\evaluate_models.py
python src\explainability.py
python -m streamlit run dashboard\app.py
```

## 19. تسلسل بناء وتشغيل EXE

```cmd
cd power_grid_cyber_ai_full
.venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
build_exe.bat
dist\PowerGridCyberAI\PowerGridCyberAI.exe
```

## 20. حل المشكلات الشائعة

### خطأ: Target not found

استخدم:

```cmd
python src\train_models.py --target marker
```

### خطأ: لا توجد ملفات CSV

تحقق من:

```cmd
dir data\raw
```

أو:

```cmd
dir data\selected
```

### خطأ: ملفات النموذج غير موجودة

أعد التدريب:

```cmd
python src\train_models.py --target marker
```

### خطأ: صفحة Real Prediction لا ترى النموذج

تأكد من وجود:

```text
models\random_forest.joblib
models\imputer.joblib
models\label_encoder.joblib
models\feature_order.json
```

وتأكد أن صفحة `2_Real_Prediction.py` تستخدم:

```python
ROOT = Path(__file__).resolve().parents[2]
```

### خطأ: صفحة إضافية لا تظهر

تأكد من وجودها هنا بالضبط:

```text
dashboard\pages\2_Real_Prediction.py
```

ثم أوقف التطبيق وشغّله من جديد:

```cmd
python -m streamlit run dashboard\app.py
```

### خطأ: Streamlit يعمل على صفحة قديمة

أوقفه:

```text
Ctrl + C
```

ثم شغّله مرة أخرى:

```cmd
python -m streamlit run dashboard\app.py
```

## 21. تنبيه أمني وتشغيلي

هذا المشروع يعمل على بيانات تاريخية أو محاكاة فقط. لا تربطه بشبكة كهرباء حقيقية، ولا تستخدم Risk Score وحده لتنفيذ قرار تشغيلي. يجب مراجعة النتائج من شخص مختص قبل أي إجراء.
