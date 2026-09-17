from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

st.subheader("Real Prediction")
st.caption("تحليل صف حقيقي من ملف CSV باستخدام النموذج المحفوظ")

model_path = MODELS / "random_forest.joblib"
imputer_path = MODELS / "imputer.joblib"
encoder_path = MODELS / "label_encoder.joblib"
features_path = MODELS / "feature_order.json"

required = [model_path, imputer_path, encoder_path, features_path]
missing = [str(path) for path in required if not path.exists()]

if missing:
    st.error("ملفات النموذج غير موجودة. شغّل التدريب أولاً.")
    st.code("python src\\train_models.py --target marker")
    st.stop()

model = load(model_path)
imputer = load(imputer_path)
encoder = load(encoder_path)
feature_order = json.loads(features_path.read_text(encoding="utf-8"))

uploaded = st.file_uploader("ارفع ملف CSV يحتوي على خصائص القياس", type=["csv"])

if uploaded is None:
    st.info("ارفع ملف CSV للبدء. يجب أن يحتوي على الخصائص المستخدمة أثناء التدريب.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"تعذر قراءة الملف: {exc}")
    st.stop()

st.write(f"عدد الصفوف: {len(data):,} | عدد الأعمدة: {len(data.columns):,}")

missing_features = [column for column in feature_order if column not in data.columns]
if missing_features:
    st.error(f"الملف يفتقد {len(missing_features)} خاصية مطلوبة.")
    st.code("\n".join(missing_features[:30]))
    st.stop()

row_number = st.number_input(
    "رقم الصف للتحليل",
    min_value=0,
    max_value=max(0, len(data) - 1),
    value=0,
    step=1,
)

row = data.iloc[[int(row_number)]].copy()
X = row.reindex(columns=feature_order).apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X_ready = imputer.transform(X)
X_ready = np.nan_to_num(X_ready, nan=0.0, posinf=0.0, neginf=0.0)

probabilities_array = model.predict_proba(X_ready)[0]
model_classes = model.classes_.astype(int)
class_names = encoder.inverse_transform(model_classes)
probabilities = {
    str(name): float(probability)
    for name, probability in zip(class_names, probabilities_array)
}

predicted = str(class_names[int(np.argmax(probabilities_array))])
risk_score = 1.0 - probabilities.get("Normal", 0.0)
log_path = ROOT / "results" / "prediction_log.csv"

if st.button("حفظ نتيجة التنبؤ"):
    log_row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "filename": uploaded.name,
                "row_number": int(row_number),
                "predicted_class": predicted,
                "risk_score": risk_score,
                "p_normal": probabilities.get("Normal", 0.0),
                "p_fault": probabilities.get("Fault", 0.0),
                "p_attack": probabilities.get("Attack", 0.0),
            }
        ]
    )

    if log_path.exists():
        log_row.to_csv(
            log_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        log_row.to_csv(
            log_path,
            index=False,
        )

    st.success("تم حفظ نتيجة التنبؤ في prediction_log.csv")

if log_path.exists():
    st.subheader("سجل التنبؤات")

    log_df = pd.read_csv(log_path)
    st.dataframe(
        log_df.tail(20),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "تنزيل سجل التنبؤات",
        data=log_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_log.csv",
        mime="text/csv",
    )


st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("الفئة المتوقعة", predicted)
c2.metric("Risk Score", f"{risk_score:.2%}")
c3.metric("P(Attack)", f"{probabilities.get('Attack', 0.0):.2%}")

st.subheader("احتمالات النموذج")
probability_table = pd.DataFrame(
    {
        "Class": list(probabilities.keys()),
        "Probability": [f"{value:.2%}" for value in probabilities.values()],
    }
)
st.dataframe(probability_table, use_container_width=True, hide_index=True)
st.bar_chart(pd.Series(probabilities))

with st.expander("عرض الخصائص المستخدمة"):
    st.dataframe(row[feature_order].T.rename(columns={row.index[0]: "value"}), use_container_width=True)

st.warning(
    "هذه النتيجة ناتجة عن نموذج مدرّب على بيانات تاريخية/محاكاة. "
    "لا تستخدمها لإرسال أوامر إلى شبكة كهرباء حقيقية."
)
